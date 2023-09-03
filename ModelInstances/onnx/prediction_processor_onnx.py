import numpy as np
from ProcessPredictions.prediction_processorOD import PredictionProcessorOD
from cv2.dnn import NMSBoxes


class PredictionProcessorOnnx(PredictionProcessorOD):

    def extract_predictions(self, input_shape, image_shape, **args):

        box_output = args.get('box_output')
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return None

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]
        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions, input_shape, image_shape)
        #
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        # boxes = self.extract_boxes(box_predictions, input_shape, image_shape)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

        return np.array(boxes)[indices], np.array(scores)[indices], np.array(class_ids)[indices], np.array(mask_predictions)[indices]

    def extract_boxes(self, box_predictions, input_shape, image_shape):
        # Extract and rescale boxes from predictions
        boxes = box_predictions[:, :4]
        # Scale boxes to original image dimensions
        image_height, image_width = image_shape[0], image_shape[1]
        boxes = self.rescale_boxes(boxes, input_shape=input_shape, image_shape=image_shape)
        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        # boxes = np.array([scale_bbox(input_shape, image_shape, b)for b in boxes])

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, image_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, image_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, image_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, image_height)

        return boxes

