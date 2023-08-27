import cv2
import numpy as np
import onnxruntime

from ModelInstances.base_class import ObjectDetectorBase
from ModelInstances.onnx.prediction_processor_onnx import PredictionProcessorOnnx


class OnnxTFLite(ObjectDetectorBase):
    def __init__(self, model_path, classes):
        super().__init__(model_path)

        self.model_path = model_path
        # with open(self.label_map_path, 'r') as f:
        #     self.labels = [line.strip() for line in f.readlines()]
        # self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')
        self.prediction_processor = PredictionProcessorOnnx

    def _load_model(self):

        self.session = onnxruntime.InferenceSession(self.model_path)
        ###
        # Get input info
        ###
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_image_shape = model_inputs[0].shape
        self.input_image_width = self.input_image_shape[3]
        self.input_image_height = self.input_image_shape[2]

        ###
        # Get output info
        ###
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def preprocess_image(self, image):

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_image_width, self.input_image_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_data = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_data

    def detect(self, image):

        self._load_image(image)
        input_data = self.preprocess_image(self.image)

        outputs = self.session.run(self.output_names, {self.input_names[0]: input_data})


        return outputs

    ###
    # Processing Predictions
    ###

    def detect_and_process(self, image):
        outputs = self.detect(image)
        processed_predictions = PredictionProcessorOnnx(outputs[0])
        detections = processed_predictions.process_predictions(box_output=outputs[0],
                                                               input_shape=self.input_image_shape[-2:],
                                                               image_shape=image.shape[:2])

        return detections # as type Detections

