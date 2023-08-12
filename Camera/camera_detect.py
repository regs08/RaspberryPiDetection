
from RaspberryPiDetection.Camera.camera_stream import CameraStream
from RaspberryPiDetection.OnnxModel.onnx_model_instance import YOLOSeg
from RaspberryPiDetection.OnnxModel.prediction_utils import *
from RaspberryPiDetection.Payload.payload import Payload
from RaspberryPiDetection.Predictions.detections import Detections


class CameraStreamDetect(CameraStream):

    def __init__(self, model_path, classes, camera_index=0, width=640, height=480):
        super().__init__(camera_index)
        # Set the resolution to 640x480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.model = YOLOSeg(model_path=model_path)

        self.classes = classes

    def show_stream(self):
        """Displays the webcam stream with random rectangles."""
        while True:
            ret, self.frame = self.cap.read()
            if ret:
                detections = self.detect()
                payload = Payload(detections, ['person'], 0.2)
                payload.send()
                cv2.imshow('prediction', self.annotate_frame(detections))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def detect(self):

        preds = self.model.segment_objects(image=self.frame)
        detections=Detections(preds)
        return detections

    def annotate_frame(self, detections):

        annotated_frame = draw_detections(image=self.frame,
                                          boxes=detections.boxes,
                                          scores=detections.scores,
                                          class_ids=detections.class_id,
                                          mask_maps=detections.mask_maps,
                                          )
        return annotated_frame

from RaspberryPiDetection.Config.default_config import default_config

if __name__ == '__main__':
    class_list = default_config['class_lists']['COCO']

    model_path = "/Users/cole/PycharmProjects/kivyTutorial/repo/App/Models/yolov8n-seg.onnx"
    stream = CameraStreamDetect(model_path=model_path, classes=class_list)
    stream.show_stream()
