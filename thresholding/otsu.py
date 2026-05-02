import numpy as np
from base import ThresholdMethod, ProcessedResult


class OtsuThreshold(ThresholdMethod):
    """
    Otsu's method: find threshold t that maximizes inter-class variance.
    sigma_b^2 = w0 * w1 * (mu0 - mu1)^2
    """
    def apply(self, gray: np.ndarray) -> ProcessedResult:
        t = self._find_threshold(gray)
        binary = (gray > t).astype(np.uint8) * 255
        return ProcessedResult("Otsu Thresholding", binary)

    def _find_threshold(self, gray: np.ndarray) -> int:
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        total = gray.size
        prob = hist / total

        best_t, best_var = 0, 0.0

        for t in range(256):
            w0 = prob[:t].sum()
            w1 = prob[t:].sum()
            if w0 == 0 or w1 == 0:
                continue
            mu0 = (np.arange(t) * prob[:t]).sum() / w0
            mu1 = (np.arange(t, 256) * prob[t:]).sum() / w1
            sigma_b = w0 * w1 * (mu0 - mu1) ** 2
            if sigma_b > best_var:
                best_var = sigma_b
                best_t = t

        return best_t
