import numpy as np
from base import ThresholdMethod, ProcessedResult


class SpectralThreshold(ThresholdMethod):
    """
    Multi-level thresholding: repeatedly apply Otsu on each sub-region
    to find n_classes-1 thresholds (more than 2 modes).
    """
    def __init__(self, n_classes: int = 3):
        self.n_classes = n_classes

    def apply(self, gray: np.ndarray) -> ProcessedResult:
        thresholds = self._find_thresholds(gray, self.n_classes)
        regions = np.digitize(gray, thresholds)  # assigns 0..n_classes-1
        result = (regions / (self.n_classes - 1) * 255).astype(np.uint8)
        return ProcessedResult(f"Spectral Thresholding ({self.n_classes} classes)", result)

    def _otsu_on_range(self, gray: np.ndarray, lo: int, hi: int) -> int:
        mask = (gray >= lo) & (gray < hi)
        if mask.sum() == 0:
            return (lo + hi) // 2
        sub = gray[mask]
        hist, _ = np.histogram(sub.flatten(), bins=hi - lo, range=(lo, hi))
        prob = hist / hist.sum()
        best_t, best_var = lo, 0.0
        for i in range(len(prob)):
            t = lo + i
            w0 = prob[:i].sum()
            w1 = prob[i:].sum()
            if w0 == 0 or w1 == 0:
                continue
            mu0 = (np.arange(i) * prob[:i]).sum() / w0
            mu1 = (np.arange(i, len(prob)) * prob[i:]).sum() / w1
            sigma_b = w0 * w1 * (mu0 - mu1) ** 2
            if sigma_b > best_var:
                best_var = sigma_b
                best_t = t
        return best_t

    def _find_thresholds(self, gray: np.ndarray, n_classes: int) -> list:
        boundaries = [0, 256]
        thresholds = []
        for _ in range(n_classes - 1):
            best_t, best_var = 0, -1
            best_seg = (0, 0)
            for i in range(len(boundaries) - 1):
                lo, hi = boundaries[i], boundaries[i + 1]
                t = self._otsu_on_range(gray, lo, hi)
                mask = (gray >= lo) & (gray < hi)
                if mask.sum() == 0:
                    continue
                sub = gray[mask]
                w0 = ((sub < t).sum()) / sub.size
                w1 = ((sub >= t).sum()) / sub.size
                mu0 = sub[sub < t].mean() if w0 > 0 else 0
                mu1 = sub[sub >= t].mean() if w1 > 0 else 0
                var = w0 * w1 * (mu0 - mu1) ** 2
                if var > best_var:
                    best_var = var
                    best_t = t
                    best_seg = (lo, hi)
            if best_t not in thresholds:
                thresholds.append(best_t)
                lo, hi = best_seg
                boundaries.remove(lo) if lo in boundaries else None
                boundaries.remove(hi) if hi in boundaries else None
                boundaries.extend([lo, best_t, hi])
                boundaries = sorted(set(boundaries))
        return sorted(thresholds)
