import cv2
import utils
from tflite_support.task import core, processor, vision
from base_model import BaseModel


class ObjectDetector(BaseModel):
        def initialize_model(self):
            """Initialize the object detector."""
            try:
                self.base_options = core.BaseOptions(
                    file_name=self.model_path, use_coral=self.enable_edgetpu, num_threads=self.num_threads)
                self.detection_options = processor.DetectionOptions(
                    max_results=self.max_results, score_threshold=0.3)
                self.options = vision.ObjectDetectorOptions(
                    base_options=self.base_options, detection_options=self.detection_options)
                self.detector = vision.ObjectDetector.create_from_options(self.options)
            except Exception as e:
                print(f"Error initializing the object detector: {e}")
                raise

        def process_frame(self, frame):
            """Run object detection and annotate the frame."""
            frame = cv2.resize(frame, self.web_dim, interpolation=cv2.INTER_AREA)
            input_tensor = vision.TensorImage.create_from_array(frame)
            detection_result = self.detector.detect(input_tensor)
            annotated_frame = utils.visualize(frame, detection_result)  # You would implement this function
            self.calculate_and_show_fps(annotated_frame)
            return annotated_frame



