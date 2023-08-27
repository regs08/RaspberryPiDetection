from Camera.camera_stream import CameraStream
import cv2
import supervision as sv

class CameraStreamDetect(CameraStream):

    def __init__(self, model, camera_index=0, width=800, height=640):
        super().__init__(camera_index)
        # Set the resolution to 640x480
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.model = model

    def show_stream(self):
        while True:
            ret, self.frame = self.cap.read()
            if ret:
                # Get detections from Model and then process them returning a Detections object
                detections = self.model.detect_and_process(image=self.frame)

                box_annotator = sv.BoxAnnotator()
                if len(detections) > 0:
                    annotated_frame = box_annotator.annotate(self.frame, detections)
                # payload = Payload(detections, ['person'], 0.2)
                # payload.send()
                # Annotate frame
                    cv2.imshow('prediction', annotated_frame)
                else: cv2.imshow('prediction', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()



def draw_rectangles(image, boxes):
    for box in boxes:
        # Retrieve the bounding box coordinates
        x_min, y_min, x_max, y_max = box.astype(int)
        # Draw the rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return image