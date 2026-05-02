import numpy as np
from base import ThresholdMethod, ProcessedResult


class OptimalThreshold(ThresholdMethod):
    def apply(self, gray: np.ndarray) -> ProcessedResult:
        t = self._find_threshold(gray)
        binary = (gray > t).astype(np.uint8) * 255
        return ProcessedResult("Optimal Thresholding", binary)

    def _find_threshold(self, gray: np.ndarray, tol: int = 1) -> float:
        t = gray.mean()
        while True:
            lo = gray[gray <= t].mean() if (gray <= t).any() else 0
            hi = gray[gray > t].mean() if (gray > t).any() else 255
            t_new = (lo + hi) / 2
            if abs(t_new - t) < tol:
                return t_new
            t = t_new
