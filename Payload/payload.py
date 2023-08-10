from datetime import datetime


class Payload:
    def __init__(self, detections, target_classes, conf):
        """
        Initialize the payload with the detections object.

        :param detections: A list of detections.
        :param target_classes: A list of target class names.
        :param conf: Confidence threshold.
        """
        self.detections = detections
        self.target_classes = target_classes
        self.conf = conf

    def format_detection(self, detection):
        """
        Format a single detection into a string with the class name and confidence, including a timestamp.

        :param detection: A single detection object.
        :return: A formatted string or None if the detection doesn't match the criteria.
        """
        confidence = detection.confidence
        class_id = detection.class_id
        class_name = self.target_classes[class_id]

        if confidence >= self.conf:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"{class_name} {confidence:.2f} at {timestamp}"
        return None

    def get_formatted_detections(self):
        """
        Format all detections into a list of strings based on target classes and confidence.

        :return: A list of formatted strings.
        """
        formatted = [self.format_detection(det) for det in self.detections]
        return [d for d in formatted if d]

    def send(self):
        print(self.get_formatted_detections())

# Mocking the detections and usage:
class Detection:
    def __init__(self, confidence, class_id):
        self.confidence = confidence
        self.class_id = class_id


# For this example, consider 0 represents "person" and 1 represents "phone".
detections = [Detection(0.95, 0), Detection(0.87, 1), Detection(0.78, 0), Detection(0.60, 1)]
target_classes = ["person", "phone"]
conf = 0.80

# Usage
payload = Payload(detections, target_classes, conf)
formatted_detections = payload.get_formatted_detections()


payload.send()