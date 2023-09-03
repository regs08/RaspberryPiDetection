from ProcessPredictions.base_class import PredictionProcessor
from supervision import Detections
import numpy as np


class PredictionProcessorENet(PredictionProcessor):

    def extract_predictions(self, outputs):
        # Get the classification result.
        class_id = np.argmax(outputs)
        confidence = outputs[class_id]
        print(confidence)
        print('class_id from extract', class_id)
        if confidence > self.conf_threshold * 100:
            return class_id, confidence
        else:
            return None

    def process_predictions(self, outputs):
        """
        note have to do hacky fix for using detections, older versions - which arre more compatible with raspi - dont
        have the Classification class
        :param outputs:
        :return:
        """

        preds = self.extract_predictions(outputs)
        if preds:
            class_id, confidence = preds
            return Detections(xyxy=np.zeros((1,4)),
                                    class_id=np.array([class_id]),
                                    confidence=np.array([confidence]),
                                    )

        return Detections.empty()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)