import numpy as np
from base import SegmentationMethod, ProcessedResult

PALETTE = np.array([
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0]
], dtype=np.uint8)


class KMeansSegmentation(SegmentationMethod):
    """
    K-Means from scratch:
    1. Pick k random pixels as initial centroids
    2. Assign each pixel to nearest centroid (Euclidean distance)
    3. Recompute centroids as mean of assigned pixels
    4. Repeat until centroids don't change
    """
    def __init__(self, k: int = 4, max_iters: int = 20):
        self.k = k
        self.max_iters = max_iters

    def apply(self, image: np.ndarray) -> ProcessedResult:
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3).astype(np.float32)

        centroids = pixels[np.random.choice(len(pixels), self.k, replace=False)]

        for _ in range(self.max_iters):
            # Distance from each pixel to each centroid: shape (n_pixels, k)
            dists = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
            labels = dists.argmin(axis=1)

            new_centroids = np.array([
                pixels[labels == i].mean(axis=0) if (labels == i).any() else centroids[i]
                for i in range(self.k)
            ])

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        result = PALETTE[labels % len(PALETTE)].reshape(h, w, 3)
        return ProcessedResult(f"K-Means (k={self.k})", result)
