import numpy as np
from base import SegmentationMethod, ProcessedResult


class MeanShiftSegmentation(SegmentationMethod):
    """
    Mean Shift from scratch:
    1. Downsample pixels for speed
    2. For each sample point, shift it toward the local mean within bandwidth h
    3. Repeat until convergence → each point settles at a mode (peak)
    4. Group settled points that are close → cluster labels
    5. Assign full-image pixels to nearest mode
    """
    def __init__(self, bandwidth: int = 30, sample_size: int = 500, max_iters: int = 20):
        self.bandwidth = bandwidth
        self.sample_size = sample_size
        self.max_iters = max_iters

    def apply(self, image: np.ndarray) -> ProcessedResult:
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3).astype(np.float32)

        idx = np.random.choice(len(pixels), min(self.sample_size, len(pixels)), replace=False)
        sample = pixels[idx].copy()

        modes = self._find_modes(sample)
        labels_sample = self._assign_to_modes(sample, modes)

        # Assign all pixels to nearest mode
        labels_full = self._assign_to_modes(pixels, modes)

        np.random.seed(3)
        colors = np.random.randint(30, 220, (len(modes), 3), dtype=np.uint8)
        result = colors[labels_full].reshape(h, w, 3)
        return ProcessedResult("Mean Shift", result)

    def _find_modes(self, points: np.ndarray) -> np.ndarray:
        shifted = points.copy()
        bw2 = self.bandwidth ** 2

        for _ in range(self.max_iters):
            new_shifted = np.zeros_like(shifted)
            for i, pt in enumerate(shifted):
                dists2 = ((points - pt) ** 2).sum(axis=1)
                in_window = dists2 <= bw2
                if in_window.any():
                    new_shifted[i] = points[in_window].mean(axis=0)
                else:
                    new_shifted[i] = pt
            if np.allclose(shifted, new_shifted, atol=0.5):
                break
            shifted = new_shifted

        return self._merge_modes(shifted)

    def _merge_modes(self, shifted: np.ndarray) -> np.ndarray:
        modes = []
        for pt in shifted:
            if not modes or all(np.linalg.norm(pt - m) > self.bandwidth * 0.5 for m in modes):
                modes.append(pt)
        return np.array(modes)

    def _assign_to_modes(self, points: np.ndarray, modes: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(points[:, None] - modes[None, :], axis=2)
        return dists.argmin(axis=1)
