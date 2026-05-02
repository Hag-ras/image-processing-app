import numpy as np
from base import SegmentationMethod, ProcessedResult


class AgglomerativeSegmentation(SegmentationMethod):
    """
    Agglomerative clustering from scratch (bottom-up):
    1. Downsample pixels for speed
    2. Each pixel starts as its own cluster
    3. Repeatedly merge the two closest clusters (by centroid distance)
    4. Stop when n_clusters remain
    5. Assign full image pixels to nearest centroid
    """
    def __init__(self, n_clusters: int = 5, sample_size: int = 300):
        self.n_clusters = n_clusters
        self.sample_size = sample_size  # work on a small sample for speed

    def apply(self, image: np.ndarray) -> ProcessedResult:
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3).astype(np.float32)

        # Sample for clustering
        idx = np.random.choice(len(pixels), min(self.sample_size, len(pixels)), replace=False)
        sample = pixels[idx]

        centroids = self._cluster(sample, self.n_clusters)

        # Assign every pixel to nearest centroid
        dists = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
        labels = dists.argmin(axis=1)

        np.random.seed(7)
        colors = np.random.randint(50, 230, (self.n_clusters, 3), dtype=np.uint8)
        result = colors[labels].reshape(h, w, 3)
        return ProcessedResult(f"Agglomerative (n={self.n_clusters})", result)

    def _cluster(self, points: np.ndarray, n_clusters: int) -> np.ndarray:
        # Start: each point is its own cluster
        clusters = {i: [points[i]] for i in range(len(points))}

        while len(clusters) > n_clusters:
            ids = list(clusters.keys())
            centroids = {i: np.mean(clusters[i], axis=0) for i in ids}

            # Find the two closest clusters by centroid distance
            best_dist = np.inf
            merge_a, merge_b = ids[0], ids[1]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    d = np.linalg.norm(centroids[ids[i]] - centroids[ids[j]])
                    if d < best_dist:
                        best_dist = d
                        merge_a, merge_b = ids[i], ids[j]

            # Merge b into a
            clusters[merge_a].extend(clusters[merge_b])
            del clusters[merge_b]

        return np.array([np.mean(clusters[i], axis=0) for i in clusters])
