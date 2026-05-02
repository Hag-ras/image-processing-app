from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class ProcessedResult:
    title: str
    image: np.ndarray  # uint8, displayable


class ThresholdMethod(ABC):
    @abstractmethod
    def apply(self, gray: np.ndarray) -> ProcessedResult: ...


class SegmentationMethod(ABC):
    @abstractmethod
    def apply(self, image: np.ndarray) -> ProcessedResult: ...
