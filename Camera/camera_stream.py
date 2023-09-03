import cv2
import time


class CameraStream:
    def __init__(self, camera_index=0):
        """Initializes the webcam stream."""
        self.cap = cv2.VideoCapture(camera_index)
        self.last_time = None

        if not self.cap.isOpened():
            print("Could not open webcam!")
            exit()

        # Grab the resolution of the stream
        ret, self.frame = self.cap.read()
        if ret:
            self.stream_width = self.frame.shape[1]
            self.stream_height = self.frame.shape[0]
        else:
            print("Could not read frame!")
            exit()

    def show_fps(self, frame):
        """Calculates and displays FPS on the frame."""
        if self.last_time is None:
            self.last_time = time.time()
            fps = 0
        else:
            curr_time = time.time()
            fps = 1 / (curr_time - self.last_time)
            self.last_time = curr_time

        # Put FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def show_stream(self):
        while True:
            ret, self.frame = self.cap.read()
            if ret:
                self.show_fps(self.frame)  # Display FPS
                cv2.imshow('Camera Stream', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        """Releases the video capture object and closes open windows."""
        self.cap.release()
        cv2.destroyAllWindows()
