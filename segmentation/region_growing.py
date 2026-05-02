import numpy as np
from collections import deque
from base import SegmentationMethod, ProcessedResult


class RegionGrowingSegmentation(SegmentationMethod):
    """
    Region Growing from scratch:
    1. Pick seed pixels on a grid
    2. BFS from each seed: add neighbor if |intensity diff| < threshold
    3. Color each grown region differently
    """
    def __init__(self, seed_step: int = 30, tolerance: int = 15):
        self.seed_step = seed_step    # distance between seed points
        self.tolerance = tolerance    # max intensity diff to join region

    def apply(self, image: np.ndarray) -> ProcessedResult:
        gray = (0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]).astype(np.uint8)
        h, w = gray.shape
        labels = np.full((h, w), -1, dtype=int)
        region_id = 0

        seeds = [
            (r, c)
            for r in range(self.seed_step // 2, h, self.seed_step)
            for c in range(self.seed_step // 2, w, self.seed_step)
        ]

        for seed in seeds:
            if labels[seed] != -1:
                continue
            self._grow(gray, labels, seed, region_id, self.tolerance)
            region_id += 1

        result = self._color_labels(labels, region_id)
        return ProcessedResult("Region Growing", result)

    def _grow(self, gray, labels, seed, region_id, tolerance):
        h, w = gray.shape
        queue = deque([seed])
        seed_val = int(gray[seed])
        labels[seed] = region_id

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                    if abs(int(gray[nr, nc]) - seed_val) <= tolerance:
                        labels[nr, nc] = region_id
                        queue.append((nr, nc))

    def _color_labels(self, labels, n_regions):
        np.random.seed(42)
        colors = np.random.randint(40, 230, (n_regions + 1, 3), dtype=np.uint8)
        label_img = np.where(labels == -1, n_regions, labels)
        return colors[label_img]
