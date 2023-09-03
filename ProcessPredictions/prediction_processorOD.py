"""
Our base class for using object detection models such as onnx and tflite. Masks not supported
"""

from abc import abstractmethod
from dataclasses import dataclass, field
import numpy as np
import supervision as sv

from ProcessPredictions.base_class import PredictionProcessor


@dataclass
class PredictionProcessorOD(PredictionProcessor):
    mask: np.ndarray = field(default_factory=list)
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
        preds = self.extract_predictions(input_shape, image_shape, **args)
        if preds:
            boxes, scores, class_ids, mask = preds
            detections = sv.Detections(xyxy=boxes,
                                        confidence=np.array(scores),
                                        class_id=np.array(class_ids),
                                       tracker_id=None
                                        )
                                        #mask=mask)
            return detections
        return sv.Detections.empty()


    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        #print('input size', input_shape, image_shape)
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

