"""
Want the base class to process the predictions of the model and then return a detections based object
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

from Predictions.core import Detections
import supervision as sv
@dataclass
class PredictionProcessor(ABC):
    outputs: np.ndarray
    mask: np.ndarray = field(default_factory=list)  #todo intialize as empty np
    iou_threshold = float = 0.7
    conf_threshold: float = 0.5
    class_threshold: float = 0.5
    label_size: float = 0.5
    boxes: np.ndarray = field(default_factory=list)
    box_confidences: np.ndarray = field(default_factory=list)
    class_ids: np.ndarray = field(default_factory=list)
    score: np.ndarray = field(default_factory=list)
    num_masks: int = 0


    @abstractmethod
    def extract_predictions(self, input_shape, image_shape, **args):
        pass

    def process_predictions(self, input_shape, image_shape, **args):

        # Extracts predicionts after applying nms stores as class variables

        boxes, scores, class_ids, mask = self.extract_predictions(input_shape, image_shape, **args)

        print(scores.shape, class_ids.shape)
        detections = sv.Detections(xyxy=boxes,
                                    confidence=np.array(scores),
                                    class_id=np.array(class_ids))
                                    #mask=mask)

        return detections


    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        print('input size', input_shape, image_shape)
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes

    @staticmethod
    def xywh2xyxy(x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    @staticmethod
    def compute_iou(box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

