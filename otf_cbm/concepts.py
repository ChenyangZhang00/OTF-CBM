from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ConceptBank:
    global_embeddings: torch.Tensor
    class_embeddings: list[torch.Tensor]
    class_names: list[str]
    global_texts: list[str]

    @property
    def num_concepts(self) -> int:
        return int(self.global_embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.global_embeddings.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.class_embeddings)

    def validate(self) -> None:
        if self.global_embeddings.ndim != 2 or not len(self.global_embeddings):
            raise ValueError("global concept embeddings must have shape [M, D], M > 0")
        if len(self.global_texts) not in (0, self.num_concepts):
            raise ValueError(
                "global_texts and global_embeddings have different lengths"
            )
        if len(self.class_names) != self.num_classes:
            raise ValueError("class_names and class_embeddings have different lengths")
        for index, embedding in enumerate(self.class_embeddings):
            if embedding.ndim != 2 or embedding.shape[0] == 0:
                raise ValueError(f"class {index} has no concept embeddings")
            if embedding.shape[1] != self.embedding_dim:
                raise ValueError(f"class {index} has the wrong embedding dimension")

    def sample_for_labels(
        self,
        labels: torch.Tensor,
        count: int,
        device: torch.device,
    ) -> torch.Tensor:
        samples = []
        for label in labels.tolist():
            pool = self.class_embeddings[int(label)]
            size = pool.shape[0]
            if size >= count:
                indices = torch.randperm(size)[:count]
            else:
                indices = torch.cat(
                    [torch.arange(size), torch.randint(size, (count - size,))]
                )
            samples.append(pool[indices])
        return torch.stack(samples).to(device=device, dtype=torch.float32)

    @classmethod
    def load(cls, path: str | Path) -> "ConceptBank":
        payload: Any = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(
                "Concept bank must be a dictionary saved by prepare_concepts.py"
            )
        bank = cls(
            global_embeddings=torch.as_tensor(payload["global_embeddings"]).float(),
            class_embeddings=[
                torch.as_tensor(item).float() for item in payload["class_embeddings"]
            ],
            class_names=[str(item) for item in payload["class_names"]],
            global_texts=[str(item) for item in payload.get("global_texts", [])],
        )
        bank.validate()
        return bank
