from Camera.camera_stream import CameraStream
import cv2
import supervision as sv
# todo will need to create a seperate detect for OD and classification

class CameraStreamDetect(CameraStream):
    def __init__(self, model, camera_index=0):
        super().__init__(camera_index)

        self.model = model

    def show_stream(self):
        while True:
            ret, self.frame = self.cap.read()
            if ret:
                # Get detections from Model and then process them returning a Detections object
                detections, labels = self.model.detect_and_process(image=self.frame)

                box_annotator = sv.BoxAnnotator()
                display_frame = self.frame

                if len(detections) > 0:
                    # Annotate frame
                    display_frame = box_annotator.annotate(self.frame, detections, labels=labels)

                # payload = Payload(detections, ['person'], 0.2)
                # payload.send()
                draw_labels_with_confidence(display_frame, labels, detections.confidence)
                self.show_fps(display_frame)
                cv2.imshow('prediction', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()



def draw_labels_with_confidence(image, labels, confidences):
    """
    Draw labels with their corresponding confidences on the top-right corner of the image.

    Parameters:
    - image: The image on which to draw the text.
    - labels: List of label texts.
    - confidences: List of confidence values.
    """
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    font_thickness = 2

    # Calculate image width
    image_width = image.shape[1]

    # Initial y-position to start drawing text
    y = 50

    for i, (label, confidence) in enumerate(zip(labels, confidences)):
        # Text to draw
        text = f"{label} ({confidence:.2f})"

        # Get the size of the text box
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Calculate x position
        x = image_width - text_width - 10  # 10 is a small margin from the right edge

        # Draw the text
        cv2.putText(image, text, (x, y + i * 40), font, font_scale, font_color, font_thickness)


