# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to display the pose detection results."""

import cv2
import numpy as np
from tflite_support.task import processor
import supervision as sv
from tensorflow_lite_support.python.task.processor.proto import bounding_box_pb2

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red


def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.

  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.

  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box

    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (_MARGIN + bbox.origin_x,
                     _MARGIN + _ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

  return image

def bbox_to_numpy_xyxy(bbox: bounding_box_pb2.BoundingBox):
    # Access the bounding box values from the BoundingBox object
    xmin = bbox.origin_x
    ymin = bbox.origin_y
    xmax = xmin + bbox.width
    ymax = ymin + bbox.height

    # Create a NumPy array [x1, y1, x2, y2]
    bbox_np = np.array([xmin, ymin, xmax, ymax])

    return bbox_np

def extract_detection_data(detection_result: processor.DetectionResult):
  bboxes = []
  labels = []
  confs = []

  for detection in detection_result.detections:
    # Draw bounding_box
    bboxes.append(bbox_to_numpy_xyxy(detection.bounding_box))

    # Draw label and score
    category = detection.categories[0]
    labels.append(category.category_name)
    confs.append(round(category.score, 2))

  return bboxes, labels, confs


def annotate_frame(image, detection_result: processor.DetectionResult):

  bboxes, labels, confs = extract_detection_data(detection_result)
  # Checking for detections
  if len(bboxes) > 0:
    bboxes = np.array(bboxes)

    detections = sv.Detections(xyxy=np.array(bboxes),
                               confidence=np.array(confs))

    # Annotating Frame
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(image, detections, labels=labels)
    return annotated_frame

  return image


def extract_classification_data():
    pass
