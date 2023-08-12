import numpy as np
from typing import Any, Iterator, List, Optional, Tuple, Union


class Detections:
    def __init__(self, preds):
        """
        Initialize the payload with the detections object.

        :param detections: A list of detections.
        :param target_classes: A list of target class names.
        :param conf: Confidence threshold.
        """
        self.boxes = preds[0]
        self.scores = preds[1]
        self.class_id = preds[2]
        self.mask_maps = preds[3]
        self.tracker_id=None
        self.length = len(self.boxes)  # Calculate the length of boxes

    def __len__(self):
        return self.length

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[float],
            Optional[int],
            Optional[int],
        ]
    ]:
        """
        Iterates over the Detections object and yield a tuple of `(xyxy, mask, confidence, class_id, tracker_id)` for each detection.
        """
        for i in range(len(self.boxes)):
            yield (
                self.boxes[i],
                self.mask_maps[i] if self.mask_maps is not None else None,
                self.scores[i] if self.scores is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None,
            )

