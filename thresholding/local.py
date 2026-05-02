import numpy as np
from base import ThresholdMethod, ProcessedResult


class LocalThreshold(ThresholdMethod):
    """
    Local (adaptive) thresholding:
    Threshold each pixel = mean of its block_size x block_size neighborhood.
    Uses an integral image for efficient computation.
    """
    def __init__(self, block_size: int = 51, offset: int = 10):
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.offset = offset

    def apply(self, gray: np.ndarray) -> ProcessedResult:
        local_mean = self._box_mean(gray, self.block_size)
        binary = (gray > local_mean - self.offset).astype(np.uint8) * 255
        return ProcessedResult("Local Thresholding", binary)

    def _box_mean(self, gray: np.ndarray, k: int) -> np.ndarray:
        pad = k // 2
        padded = np.pad(gray.astype(np.float64), pad, mode='reflect')
        # Build integral image (prefix sum)
        integral = padded.cumsum(axis=0).cumsum(axis=1)
        # Add one-row/col of zeros at top and left for clean indexing
        integral = np.pad(integral, ((1, 0), (1, 0)), mode='constant')

        h, w = gray.shape
        # For pixel (i,j) in original: its neighborhood in padded is rows [i, i+k), cols [j, j+k)
        # In integral (with the extra border): rows [i, i+k], cols [j, j+k]
        r = np.arange(h)[:, None]
        c = np.arange(w)[None, :]
        box_sum = (integral[r + k, c + k]
                 - integral[r,     c + k]
                 - integral[r + k, c    ]
                 + integral[r,     c    ])
        return (box_sum / (k * k)).astype(np.float32)
