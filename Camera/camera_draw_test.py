import cv2
from Camera.camera_stream import CameraStream

import random


class CameraStreamTest(CameraStream):
    def __init__(self, camera_index=0):
        super().__init__(camera_index)

    def draw_random_rectangle(self, color=(0, 255, 0), thickness=2):
        """Draws a rectangle at a random location and of a random size in the frame."""
        max_width = self.stream_width
        max_height = self.stream_height

        width = random.randint(1, max_width)
        height = random.randint(1, max_height)

        xmin = random.randint(0, max_width - width)
        ymin = random.randint(0, max_height - height)
        xmax = xmin + width
        ymax = ymin + height

        cv2.rectangle(self.frame, (xmin, ymin), (xmax, ymax), color, thickness)

    def show_stream(self):
        """Displays the webcam stream with random rectangles."""
        while True:
            ret, self.frame = self.cap.read()
            if ret:
                self.draw_random_rectangle()
                cv2.imshow('Camera Stream with Random Rectangle', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

if __name__ == '__main__':
    stream = CameraStreamTest()
    stream.show_stream()


