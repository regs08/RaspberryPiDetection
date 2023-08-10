import cv2
from RaspberryPiDetection.Camera.camera_stream import CameraStream
from RaspberryPiDetection.OnnxModel.onnx_model_instance import YOLOSeg
from supervision import Detections, BoxAnnotator, MaskAnnotator

# Should just have it take a config file load it it etc ...
class CameraStreamDetect(CameraStream):

    def __init__(self, model_path, classes, camera_index=0, width=640, height=480):
        super().__init__(camera_index)
        # Set the resolution to 640x480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.model = YOLOSeg(model_path=model_path)
        self.box_annotator = BoxAnnotator()
        self.mask_annotator = MaskAnnotator()

        self.classes = classes

    def show_stream(self):
        """Displays the webcam stream with random rectangles."""
        while True:
            ret, self.frame = self.cap.read()
            if ret:

                cv2.imshow('prediction', self.annotate_frame())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def detect(self):

        preds = self.model.segment_objects(image=self.frame)
        detections = self.create_detections(preds)

        return detections

    def annotate_frame(self):
        detections = self.detect()
        labels = [f"{self.classes[class_id]} {float(confidence):0.2f}"
                  for _, _, confidence, class_id, _ in detections]

        annotated_frame = self.box_annotator.annotate(
            scene=self.frame.copy(),
            detections=detections,
            labels=labels
        )

        annotated_frame_masks = self.mask_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        return annotated_frame_masks

    # to go in prediction processor
    @staticmethod
    def create_detections(preds):

        detections = Detections.empty()
        detections.xyxy = preds[0]
        detections.confidence = preds[1]
        detections.class_id = preds[2]
        detections.mask = preds[3]

        return detections

from RaspberryPiDetection.Config.default_config import default_config

if __name__ == '__main__':
    class_list = default_config['class_lists']['COCO']

    model_path = "/Users/cole/PycharmProjects/kivyTutorial/repo/App/Models/yolov8n-seg.onnx"
    stream = CameraStreamDetect(model_path=model_path, classes=class_list)
    stream.show_stream()
