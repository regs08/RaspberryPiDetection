"""
Base class for tflite models
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2


class ObjectDetectorBase(ABC):
    def __init__(self, model_path, box_threshold=0.5, class_threshold=0.5, label_size=0.5):
        self.model_path = model_path

        self.box_threshold = box_threshold
        self.class_threshold = class_threshold
        self.label_size = label_size

        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def detect(self, image):
        pass

    def _load_image(self, input_image):
        if isinstance(input_image, str):
            self.image = cv2.imread(input_image, cv2.IMREAD_COLOR)
        elif isinstance(input_image, np.ndarray):
            self.image = input_image
        else:
            raise TypeError("Input must be either a file path (string) or a NumPy array representing an image.")

    # More common functionalities can be added here
