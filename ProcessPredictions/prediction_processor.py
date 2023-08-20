import cv2
import numpy as np

# get necessarry info from object detectorr type cast the init
#
# I think if we just have this take in the outputs from our model class it will be morre robust

class PredictionProcessor:

    def __init__(self, outputs, box_threshold=0.5, class_threshold=0.5, label_size=0.5):
        self.outputs = outputs
        self.box_threshold = box_threshold
        self.class_threshold = class_threshold
        self.label_size = label_size
        self.boxes = []
        self.box_confidences = []
        self.class_ids = []
        self.score = []

    def extract_predictions(self, img_width, img_height):
        boxes = []
        box_confidences = []
        class_ids = []
        confidence =[]
        for output in self.outputs:
            box_confidence = output[4]
            if box_confidence < self.box_threshold:
                continue

            class_id= output[5:].argmax(axis=0)
            conf = output[5:][class_id]

            if conf < self.class_threshold:
                continue

            cx, cy, w, h = output[:4] * np.array([img_width, img_height, img_width, img_height])
            xmin = round(cx - w / 2)
            ymin = round(cy - h / 2)
            xmax = round(cx + w / 2)
            ymax = round(cy + h / 2)

            boxes.append([xmin, ymin, xmax, ymax])
            box_confidences.append(box_confidence)
            class_ids.append(class_id)
            confidence.append(conf)


        indices = cv2.dnn.NMSBoxes(boxes, box_confidences, self.box_threshold, self.box_threshold - 0.1)

        for indice in indices:
            self.boxes.append(boxes[indice])
            self.class_ids.append(class_ids[indice])
            self.score = box_confidences[indice] * confidence[indice]

    def process_predictions(self, input_shape, image_shape):
        """
        applys nms and rescales predictions from tf lite output
        :param input_shape: shape expected by the model e.g 320, 320
        :param image_shape: shape of the image passed in e.g 3280, 4000
        :return:
        """
        # Extracts predicionts after applying nms stores as class variables
        self.extract_predictions(img_width=input_shape[0], img_height=input_shape[1])
        self.boxes = self.rescale_boxes(self.boxes, input_shape, image_shape)

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes

    # def annotate_boxes_labels(self, img_padded, indices):
    #     for indice in indices:
    #         xmin, ymin, xmax, ymax = self.boxes[indice[0]]
    #         class_name = self.labels[self.classes[indice[0]]]
    #         score = self.box_confidences[indice[0]] * self.class_probs[indice[0]]
    #         color = [int(c) for c in self.colors[self.classes[indice[0]]]]
    #         text_color = (255, 255, 255) if sum(color) < 144 * 3 else (0, 0, 0)
    #
    #         cv2.rectangle(img_padded, (xmin, ymin), (xmax, ymax), color, 2)
    #
    #         label = f'{class_name}: {score * 100:.2f}%'
    #         labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.label_size, 2)
    #         cv2.rectangle(img_padded,
    #                       (xmin, ymin + baseLine), (xmin + labelSize[0], ymin - baseLine - labelSize[1]),
    #                       color, cv2.FILLED)
    #         cv2.putText(img_padded, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, self.label_size, text_color, 1)
    #
    #     return img_padded

