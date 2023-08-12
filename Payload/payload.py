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

    def format_detection(self, confidence, class_id):
        """
        Format a single detection into a string with the class name and confidence, including a timestamp.

        :param detection: A single detection object.
        :return: A formatted string or None if the detection doesn't match the criteria.
        """

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
        formatted = [self.format_detection(conf, class_id) for _, _, conf, class_id, _ in self.detections]
        return [d for d in formatted if d]

    def send(self):
        pkg = self.get_formatted_detections()
        if pkg:
            print(pkg)

