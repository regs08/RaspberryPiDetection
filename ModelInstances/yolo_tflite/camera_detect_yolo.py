from ModelInstances.yolo_tflite.yolo_model import YOLOTFLite
from Camera.camera_detect import CameraStreamDetect
from Config.default_config import default_config
from ModelInstances.tf_lite_yolo import TFLiteYOLO
from ModelInstances.yolo_tflite.prediction_processor_yolo import PredictionProcessorYOLO

model_path = default_config['model_paths']['tflite-yolo-v5']
label_map_path = default_config['label_maps']['COCO']

yolo_tf_lite = TFLiteYOLO(model_path=model_path,
                          labels=default_config['label_maps']['COCO'],
                          prediction_processor=PredictionProcessorYOLO)

camera = CameraStreamDetect(model=yolo_tf_lite)
camera.show_stream()
