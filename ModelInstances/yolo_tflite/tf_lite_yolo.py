from ModelInstances.tf_lite_base import TFLite
import cv2
import numpy as np

"""
 Just need to define preprocess image. YOlO is taking in a float where efficient net is taking in an int
"""


class TFLiteYOLO(TFLite):
    def preprocess_image(self, image):

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_image_width, self.input_image_height), interpolation=cv2.INTER_AREA)
        input_data = np.expand_dims(img_resized / 255, axis=0).astype('float32')

        return input_data

    def detect_and_process(self, image):
        # Todo fix the prediction class with the constructor being called like this
        outputs = self.detect(image)
        processed_predictions = self.prediction_processor(outputs)
        # Checks for predictions
        detections = processed_predictions.process_predictions(input_shape=(self.input_image_height, self.input_image_width),
                                                               image_shape=image.shape[:2])
        return detections # as type Detections