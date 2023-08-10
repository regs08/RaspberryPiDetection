import cv2


class CameraStream:
    def __init__(self, camera_index=0):
        """Initializes the webcam stream."""
        self.cap = cv2.VideoCapture(camera_index)

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

    def show_stream(self):
        """Displays the webcam stream."""
        while True:
            ret, self.frame = self.cap.read()
            if ret:
                cv2.imshow('Camera Stream', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        """Releases the video capture object and closes open windows."""
        self.cap.release()
        cv2.destroyAllWindows()
