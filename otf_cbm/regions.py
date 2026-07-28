from __future__ import annotations

import torch

try:
    import faiss
except ImportError:  # pragma: no cover - exercised only without optional backend
    faiss = None


def _torch_kmeans(
    features: torch.Tensor,
    clusters: int,
    iterations: int,
) -> torch.Tensor:
    count = features.shape[0]
    if clusters > count:
        raise ValueError(f"Cannot form {clusters} clusters from {count} patches")
    initial = torch.linspace(0, count - 1, clusters, device=features.device).long()
    centroids = features[initial].clone()
    labels = torch.zeros(count, device=features.device, dtype=torch.long)
    for _ in range(iterations):
        distances = torch.cdist(features.float(), centroids.float())
        new_labels = distances.argmin(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(clusters):
            members = features[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(dim=0)
    return labels


def cluster_patches(
    patch_features: torch.Tensor,
    clusters: int = 16,
    iterations: int = 20,
    backend: str = "faiss",
    seed: int = 1234,
) -> torch.Tensor:
    labels = []
    for sample in patch_features.detach():
        if backend == "faiss" and faiss is not None:
            array = sample.float().cpu().numpy()
            kmeans = faiss.Kmeans(
                d=array.shape[1],
                k=clusters,
                niter=iterations,
                verbose=False,
                seed=seed,
            )
            kmeans.train(array)
            _, indices = kmeans.index.search(array, 1)
            label = torch.from_numpy(indices[:, 0]).to(sample.device)
        else:
            label = _torch_kmeans(sample, clusters, iterations)
        labels.append(label.long())
    return torch.stack(labels)


def pool_regions(
    projected_patches: torch.Tensor,
    cluster_labels: torch.Tensor,
    clusters: int,
) -> torch.Tensor:
    batch, _, dimension = projected_patches.shape
    regions = projected_patches.new_zeros(batch, clusters, dimension)
    counts = projected_patches.new_zeros(batch, clusters, 1)
    index = cluster_labels.unsqueeze(-1).expand(-1, -1, dimension)
    regions.scatter_add_(1, index, projected_patches)
    counts.scatter_add_(
        1,
        cluster_labels.unsqueeze(-1),
        torch.ones_like(cluster_labels, dtype=projected_patches.dtype).unsqueeze(-1),
    )
    return regions / counts.clamp_min(1.0)


def background_clusters(
    foreground_patches: torch.Tensor,
    cluster_labels: torch.Tensor,
    clusters: int,
    ratio_threshold: float = 0.7,
) -> torch.Tensor:
    background_votes = (~foreground_patches.bool()).float()
    background_counts = background_votes.new_zeros(background_votes.shape[0], clusters)
    total_counts = background_votes.new_zeros(background_votes.shape[0], clusters)
    background_counts.scatter_add_(1, cluster_labels, background_votes)
    total_counts.scatter_add_(1, cluster_labels, torch.ones_like(background_votes))
    ratios = background_counts / total_counts.clamp_min(1.0)
    return ratios > ratio_threshold
