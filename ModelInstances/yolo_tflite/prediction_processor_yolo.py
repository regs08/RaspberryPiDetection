import numpy as np
import cv2
from ProcessPredictions.core import PredictionProcessor


class PredictionProcessorYOLO(PredictionProcessor):

    def extract_predictions(self, input_shape, image_shape, **args):
        """
        gets the prediction input from our output. we scale the boxes and masks(optional) todo have mask functionality

        :param input_width:
        :param input_height:
        :return:
        """
        input_width, input_height = input_shape[1], input_shape[0]
        boxes = []
        scores = []
        class_ids = []
        confidence =[]
        mask=[]

        for output in self.outputs:
            box_confidence = output[4]
            if box_confidence < self.conf_threshold:
                continue

            class_id= output[5:].argmax(axis=0)
            conf = output[5:][class_id]
            if conf < self.class_threshold:
                continue

            cx, cy, w, h = output[:4] * np.array([input_width, input_height, input_width, input_height])
            xmin = round(cx - w / 2)
            ymin = round(cy - h / 2)
            xmax = round(cx + w / 2)
            ymax = round(cy + h / 2)

            boxes.append([xmin, ymin, xmax, ymax])
            scores.append(box_confidence)
            class_ids.append(class_id)
            confidence.append(conf)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.conf_threshold - 0.1)

        for indice in indices:
            self.boxes.append(boxes[indice])
            self.class_ids.append(class_ids[indice])
            self.score = scores[indice] * confidence[indice]
        self.boxes = self.rescale_boxes(self.boxes, input_shape, image_shape)
        return np.array(self.boxes), np.array([self.score]), np.array(self.class_ids), mask

    def detection_check(self):
        if len(self.boxes) == 0:
            self.boxes = np.zeros((0,4))
            self.score = np.zeros((0,1))
            self.class_ids = np.zeros((0,1))


