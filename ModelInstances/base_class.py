"""
Base class for loading models. Can take in different formated outputs such as onnx and tflite. takes in a given image
and returns a Detections object.

Uses a predicion Processor specific to the type of the model.
"""
from abc import ABC, abstractmethod
import numpy as np
import cv2
from ProcessPredictions.prediction_processorOD import PredictionProcessor
from supervision import Detections
from typing import Union, List


class ObjectDetectorBase(ABC):
    def __init__(self, model_path,
                 labels: Union[List[str], str],
                 prediction_processor: PredictionProcessor,
                 box_threshold=0.5,
                 class_threshold=0.1,
                 label_size=0.5):

        self.model_path = model_path
        self.box_threshold = box_threshold
        self.class_threshold = class_threshold
        self.label_size = label_size
        self.prediction_processor = prediction_processor

        self.labels = self.get_labels(labels)
        self.label_id_map, self.id_label_map = self.get_label_maps()
        self.colors = self.get_colors()
        self._load_model()

    def _load_image(self, input_image):
        if isinstance(input_image, str):
            self.image = cv2.imread(input_image, cv2.IMREAD_COLOR)
        elif isinstance(input_image, np.ndarray):
            self.image = input_image

        else:
            raise TypeError("Input must be either a file path (string) or a NumPy array representing an image.")
        self.image_height, self.image_width = self.image.shape[:2]

    def get_labels(self, labels: Union[List[str], str]):
        if isinstance(labels, str):
            # labels is a path (string)
            if not labels.endswith('.txt'):
                raise ValueError("File path must end with .txt")
            return self.load_label_map(labels)

        elif isinstance(labels, list):
            return labels
        else:
            raise TypeError("Invalid type for labels; must be a list of strings or a .txt file path.")

    def get_label_maps(self):
        label_id_map = {label: idx for idx, label in enumerate(self.labels)}
        id_label_map = {idx: label for label, idx in label_id_map.items()}

        return label_id_map, id_label_map


    def get_colors(self):
       return np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')

    @abstractmethod
    def _load_model(self):
        """
        loads the model and assins class vars to be used for inference and processing
        :return:
        """
        pass

    @abstractmethod
    def detect(self, image):
        """
        detects on image then returns ouputs
        :param image:
        :return:
        """
        pass

    @abstractmethod
    def preprocess_image(self, image):
        """
        Preprocess the image for the given model
        :param image:
        :return:
        """

        pass

    @abstractmethod
    def detect_and_process(self, image) -> Detections:
        """
        wrapper class that will use the model's detect method to get the output then the passed in Prediction processor
        to return a Detections object to be used for plotting.
        :param image:
        :return:
        """
        pass

    @staticmethod
    def load_label_map(filepath: str) -> list:
        if not filepath.endswith('.txt'):
            raise ValueError("File path must end with .txt")

        labels = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    labels.append(line.strip())
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filepath} not found.")

        return labels

