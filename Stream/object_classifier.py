import cv2
import utils
from tflite_support.task import core, processor, vision
from Stream.base_model import BaseModel


class ObjectDetector(BaseModel):
        def initialize_model(self):
            """Initialize the object detector."""
            try:
                self.base_options = core.BaseOptions(file_name=self.model_path)

                self.classification_options = processor.ClassificationOptions(
                    max_results=self.max_results)
                self.options = vision.ImageClassifierOptions(
                    base_options=self.base_options, classification_options=self.classification_options)
                self.classifier = vision.ImageClassifier.create_from_options(self.options)

            except Exception as e:
                print(f"Error initializing the object detector: {e}")
                raise

        def process_frame(self, frame):
            """Run object detection and annotate the frame."""
            frame = cv2.resize(frame, self.web_dim, interpolation=cv2.INTER_AREA)
            input_tensor = vision.TensorImage.create_from_array(frame)
            categories = self.classifier.classify(input_tensor)

            for idx, category in enumerate(categories.classifications[0].categories):
                category_name = category.category_name
                score = round(category.score, 2)
                result_text = category_name + ' (' + str(score) + ')'
                text_location = (_LEFT_MARGIN, (idx + 2) * _ROW_SIZE)
                cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                            _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
            annotated_frame = utils.visualize(frame, detection_result)  # You would implement this function
            self.calculate_and_show_fps(annotated_frame)
            return annotated_frame



