from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..backbones import (
    extract_dino_features,
    foreground_patch_mask,
)
from ..concepts import ConceptBank
from ..regions import background_clusters, cluster_patches, pool_regions
from ..transport import unbalanced_sinkhorn_log
from .cost import CostModel, VisualAdapter
from .flow import VelocityField


class OTFCBM(nn.Module):
    """Current two-path OTF-CBM implementation.

    During training, labels are used only to sample class-conditioned concepts
    for the auxiliary flow-matching loss. ``predict`` never consumes labels.
    """

    def __init__(
        self,
        backbone: nn.Module,
        concept_bank: ConceptBank,
        num_classes: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        model_config = config["model"]
        image_dim = int(model_config.get("image_dim", 1024))
        feature_dim = int(model_config.get("feature_dim", concept_bank.embedding_dim))
        if feature_dim != concept_bank.embedding_dim:
            raise ValueError(
                f"Config feature_dim={feature_dim}, concept bank uses "
                f"{concept_bank.embedding_dim}"
            )
        if concept_bank.num_classes != num_classes:
            raise ValueError(
                f"Concept bank has {concept_bank.num_classes} classes, "
                f"dataset has {num_classes}"
            )
        normalize_adapters = bool(model_config.get("normalize_adapters", True))

        self.backbone = backbone
        self.condition_adapter = VisualAdapter(
            in_dim=image_dim,
            out_dim=feature_dim,
            hidden_dim=int(model_config.get("adapter_hidden_dim", 512)),
            use_residual=True,
            normalize=normalize_adapters,
        )
        self.velocity_field = VelocityField(
            feature_dim=feature_dim,
            condition_dim=feature_dim,
            time_dim=int(model_config.get("time_dim", 32)),
            hidden_dim=int(model_config.get("flow_hidden_dim", 512)),
        )
        self.cost_model = CostModel(
            image_dim=image_dim,
            text_dim=feature_dim,
            adapter_hidden_dim=int(model_config.get("adapter_hidden_dim", 512)),
            num_bases=int(model_config.get("num_cost_bases", 18)),
            normalize_projection=normalize_adapters,
            use_geometry_adapter=bool(model_config.get("use_geometry_adapter", True)),
            geometry_alpha_init=float(model_config.get("geometry_alpha_init", 0.1)),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(concept_bank.num_concepts),
            nn.Linear(concept_bank.num_concepts, num_classes),
        )

        self.register_buffer(
            "global_concepts",
            concept_bank.global_embeddings.float(),
            persistent=False,
        )
        self.concept_bank = concept_bank
        self.num_classes = num_classes
        self.num_concepts = concept_bank.num_concepts
        self.feature_dim = feature_dim
        self.clusters = int(model_config.get("clusters", 16))
        self.cluster_iterations = int(model_config.get("cluster_iterations", 20))
        self.cluster_backend = str(model_config.get("cluster_backend", "faiss"))
        self.cluster_seed = int(model_config.get("cluster_seed", 1234))
        self.foreground_keep_ratio = float(
            model_config.get("foreground_keep_ratio", 0.5)
        )
        self.foreground_method = str(model_config.get("foreground_method", "attention"))
        self.background_ratio = float(model_config.get("background_ratio", 0.7))
        self.background_penalty = float(model_config.get("background_penalty", 50.0))
        self.class_concept_samples = int(model_config.get("class_concept_samples", 5))
        self.activation_chunk_size = int(model_config.get("activation_chunk_size", 8))
        self.activation_top_regions = int(model_config.get("activation_top_regions", 5))
        sinkhorn = model_config.get("sinkhorn", {})
        self.sinkhorn_epsilon = float(sinkhorn.get("epsilon", 0.01))
        self.sinkhorn_tau = float(sinkhorn.get("tau", 0.1))
        self.sinkhorn_iterations = int(sinkhorn.get("iterations", 50))

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "OTFCBM":
        super().train(mode)
        self.backbone.eval()
        return self

    def freeze_stage1_ot(self) -> None:
        """Stage 2 freezes theta only; both visual adapters remain trainable."""

        self.cost_model.freeze_ot_only()

    def _extract_regions(
        self,
        images: torch.Tensor,
        need_background: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        cls_token, patch_tokens = extract_dino_features(self.backbone, images)
        labels = cluster_patches(
            patch_tokens,
            clusters=self.clusters,
            iterations=self.cluster_iterations,
            backend=self.cluster_backend,
            seed=self.cluster_seed,
        )
        projected_patches = self.cost_model.project(patch_tokens)
        regions = pool_regions(projected_patches, labels, self.clusters)
        condition = self.condition_adapter(cls_token)

        background = None
        if need_background:
            foreground = foreground_patch_mask(
                self.backbone,
                images,
                patch_tokens,
                cls_token,
                keep_ratio=self.foreground_keep_ratio,
                method=self.foreground_method,
            )
            background = background_clusters(
                foreground,
                labels,
                self.clusters,
                ratio_threshold=self.background_ratio,
            )
        return regions, condition, background

    def _flow_matching_loss(
        self,
        regions: torch.Tensor,
        condition: torch.Tensor,
        background: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        targets = self.concept_bank.sample_for_labels(
            labels,
            count=self.class_concept_samples,
            device=regions.device,
        )
        costs = []
        for sample_regions, sample_targets in zip(regions, targets):
            cost, _ = self.cost_model.cost_from_projected(
                sample_regions, sample_targets
            )
            costs.append(cost)
        cost_batch = torch.stack(costs)
        cost_batch = cost_batch + (
            background.unsqueeze(-1).to(cost_batch.dtype) * self.background_penalty
        )

        source_mass = 1.0 - background.float()
        empty = source_mass.sum(dim=1) == 0
        source_mass[empty] = 1.0
        target_mass = torch.ones(
            regions.shape[0],
            self.class_concept_samples,
            device=regions.device,
            dtype=regions.dtype,
        )
        plan = unbalanced_sinkhorn_log(
            cost_batch,
            source_mass,
            target_mass,
            epsilon=self.sinkhorn_epsilon,
            tau=self.sinkhorn_tau,
            iterations=self.sinkhorn_iterations,
        )
        matched_indices = plan.argmax(dim=1)
        sources = torch.gather(
            regions,
            dim=1,
            index=matched_indices.unsqueeze(-1).expand(-1, -1, self.feature_dim),
        )

        source_flat = sources.reshape(-1, self.feature_dim)
        target_flat = targets.reshape(-1, self.feature_dim)
        time = torch.rand(source_flat.shape[0], 1, device=regions.device)
        path_point = (1.0 - time) * source_flat + time * target_flat
        expanded_condition = (
            condition.unsqueeze(1)
            .expand(-1, self.class_concept_samples, -1)
            .reshape(-1, self.feature_dim)
        )
        predicted = self.velocity_field(path_point, time, expanded_condition)
        target_velocity = target_flat - source_flat
        return F.mse_loss(predicted, target_velocity)

    def concept_activations(
        self,
        regions: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        score_chunks = []
        batch, region_count, _ = regions.shape
        for concepts in torch.split(
            self.global_concepts, self.activation_chunk_size, dim=0
        ):
            concept_count = concepts.shape[0]
            region_view = regions.unsqueeze(2)
            concept_view = concepts.reshape(1, 1, concept_count, self.feature_dim)
            midpoint = (region_view + concept_view) / 2.0
            true_velocity = concept_view - region_view
            expanded_condition = (
                condition[:, None, None, :]
                .expand(batch, region_count, concept_count, self.feature_dim)
                .reshape(-1, self.feature_dim)
            )
            time = torch.full(
                (batch * region_count * concept_count, 1),
                0.5,
                device=regions.device,
                dtype=regions.dtype,
            )
            predicted = self.velocity_field(
                midpoint.reshape(-1, self.feature_dim),
                time,
                expanded_condition,
            ).reshape(batch, region_count, concept_count, self.feature_dim)
            scores = -(predicted - true_velocity).square().sum(dim=-1)
            score_chunks.append(scores)

        score_matrix = torch.cat(score_chunks, dim=2)
        top_regions = min(self.activation_top_regions, score_matrix.shape[1])
        return score_matrix.topk(top_regions, dim=1).values.mean(dim=1)

    def forward_train(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        regions, condition, background = self._extract_regions(
            images, need_background=True
        )
        assert background is not None
        flow_loss = self._flow_matching_loss(regions, condition, background, labels)
        activations = self.concept_activations(regions, condition)
        return self.classifier(activations), flow_loss, activations

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_activations: bool = False,
    ):
        """Dispatch through one DDP-compatible entry point."""

        if labels is None:
            return self.predict(images, return_activations=return_activations)
        if return_activations:
            raise ValueError("return_activations is only valid without labels")
        return self.forward_train(images, labels)

    def predict(
        self,
        images: torch.Tensor,
        return_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        regions, condition, _ = self._extract_regions(images, need_background=False)
        activations = self.concept_activations(regions, condition)
        logits = self.classifier(activations)
        if return_activations:
            return logits, activations
        return logits

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only OTF-CBM parameters, excluding the frozen backbone."""

        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("backbone.")
        }
