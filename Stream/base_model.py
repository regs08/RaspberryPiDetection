from abc import ABC, abstractmethod
import cv2
import time

class BaseModel(ABC):
    """A base class for real-time vision models."""

    def __init__(self, model_path, camera_id=0, width=640, height=480, num_threads=1, max_results=3, fps_avg_frame_count=5,
                 enable_edgetpu=False):
        self.enable_edgetpu = enable_edgetpu
        self.max_results = max_results
        self.model_path = model_path
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.web_dim = (self.width, self.height)
        self.num_threads = num_threads
        self.fps = 0
        self.fps_avg_frame_count = fps_avg_frame_count
        self.counter = 0
        self.start_time = time.time()
        self.initialize_model()

    @abstractmethod
    def initialize_model(self):
        """Initialize the model. Must be overridden by subclasses."""
        pass

    @abstractmethod
    def process_frame(self, frame):
        """Process a frame and return an annotated frame. Must be overridden by subclasses."""
        pass

    def calculate_and_show_fps(self, frame):
        """Calculate and show the FPS on the frame."""
        if self.counter % self.fps_avg_frame_count == 0:
            end_time = time.time()
            self.fps = self.fps_avg_frame_count / (end_time - self.start_time)
            self.start_time = time.time()
        fps_text = f'FPS = {self.fps:.1f}'
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    def generate_frames(self):
        """Generate frames for a Flask app."""
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        while True:
            self.counter += 1
            success, frame = cap.read()
            if not success:
                print("Failed to capture image")
                break
            annotated_frame = self.process_frame(frame)
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            encoded_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')


