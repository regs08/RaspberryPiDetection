"""
 Base class for processing predictions. First we extract them from the model output then we process them in some way
  ** Note that in new versions of supervision there is a  classification class. We should be using that as well but
  the older version doesn't install well on RaspPi
"""
from abc import ABC, abstractmethod
from typing import Any
from supervision import Detections
from dataclasses import dataclass
import numpy as np

@dataclass
class PredictionProcessor(ABC):
    outputs: np.ndarray
    iou_threshold = float = 0.7
    conf_threshold: float = 0.3
    class_threshold: float = 0.3
    label_size: float = 0.5

    @abstractmethod
    def extract_predictions(self,  **args: Any):
        pass
    @abstractmethod
    def process_predictions(self, **args: Any) -> Detections:
        pass


