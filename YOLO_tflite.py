import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

from ObjectDetectionBaseClass.base_class import ObjectDetectorBase
from TFliteTest.ProcessPredictions.prediction_processor import PredictionProcessor


class YOLOTFLite(ObjectDetectorBase):
    def __init__(self, model_path, label_map_path):
        super().__init__(model_path)

        self.label_map_path = label_map_path

        with open(self.label_map_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')

    def _load_model(self):

        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.width = self.input_details[0]['shape'][2]
        self.height = self.input_details[0]['shape'][1]

    def detect(self, image):
        self._load_image(image)
        self.input_data = self.preprocess_image()

        self.interpreter.set_tensor(self.input_details[0]['index'], self.input_data)
        self.interpreter.invoke()
        outputs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        return outputs

    def preprocess_image(self):

        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        input_data = np.expand_dims(img_resized / 255, axis=0).astype('float32')

        return input_data


def draw_rectangles(image, boxes):
    for box in boxes:
        # Retrieve the bounding box coordinates
        x_min, y_min, x_max, y_max = box.astype(int)
        # Draw the rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return image


TF_LITE_MODEL = '/Users/cole/PycharmProjects/testClose/TFliteTest/lite-model_yolo-v5-tflite_tflite_model_1.tflite'
LABEL_MAP = '/Users/cole/PycharmProjects/testClose/TFliteTest/labelmap.txt'

model = YOLOTFLite(TF_LITE_MODEL, LABEL_MAP)



# Initialize video capture from the default camera (usually camera index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera is ready.")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    outputs = model.detect(frame)

    processed_predictions = PredictionProcessor(outputs)
    processed_predictions.process_predictions((model.width, model.height), model.image.shape)

    image_with_rectangles = draw_rectangles(model.image, processed_predictions.boxes)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
