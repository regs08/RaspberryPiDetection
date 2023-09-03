import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from ModelInstances.base_class import ObjectDetectorBase
from ModelInstances.yolo_tflite.prediction_processor_yolo import PredictionProcessorYOLO


class YOLOTFLite(ObjectDetectorBase):
    def __init__(self, model_path, label_map_path, prediction_processor=PredictionProcessorYOLO):
        super().__init__(model_path, prediction_processor)

        self.label_map_path = label_map_path

        with open(self.label_map_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')


    def _load_model(self):

        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_image_width = self.input_details[0]['shape'][2]
        self.input_image_height = self.input_details[0]['shape'][1]

    def preprocess_image(self, image):

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_image_width, self.input_image_height), interpolation=cv2.INTER_AREA)
        input_data = np.expand_dims(img_resized / 255, axis=0).astype('float32')

        return input_data

    def detect(self, image):

        self._load_image(image)
        # Assing our passed image to the class var
        input_data = self.preprocess_image(self.image)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        outputs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        return outputs

    ###
    # Processing Predictions
    ###

    def detect_and_process(self, image):
        outputs = self.detect(image)
        processed_predictions = self.prediction_processor(outputs)
        # Checks for predictions
        detections = processed_predictions.process_predictions(input_shape=(self.input_image_height, self.input_image_width),
                                                               image_shape=image.shape[:2])
        return detections # as type Detections



