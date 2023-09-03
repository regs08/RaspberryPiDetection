import cv2
import numpy as np
from ModelInstances.tf_lite_base import TFLite
from supervision import Detections
from typing import List, Tuple


class EfficientNetTFLite(TFLite):

    def preprocess_image(self, image):
        input_shape = self.input_details[0]['shape'][1:3]  # Shape as [Height, Width]
        image_resized = cv2.resize(image, tuple(input_shape[::-1]))  # OpenCV uses (W, H) so reverse shape to (H, W)
        input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)  # Convert to UINT8

        return input_data

    def detect_and_process(self, image) -> Tuple[Detections, List[str]]:
        outputs = self.detect(image)
        preds = self.prediction_processor(outputs)
        detections = preds.process_predictions(outputs)

        labels = []
        if detections:
            labels = [self.id_label_map[class_id] for class_id in detections.class_id]
        print(labels)
        return detections, labels
