from tflite_runtime.interpreter import Interpreter
from ModelInstances.base_class import ObjectDetectorBase

class TFLite(ObjectDetectorBase):

    def _load_model(self):

        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_image_width = self.input_details[0]['shape'][2]
        self.input_image_height = self.input_details[0]['shape'][1]


    def detect(self, image):

        self._load_image(image)
        # Assing our passed image to the class var
        input_data = self.preprocess_image(self.image)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        outputs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        return outputs







