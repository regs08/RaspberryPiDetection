from Camera.camera_detect import CameraStreamDetect
from Config.default_config import default_config
from ModelInstances.yolo_tflite.tf_lite_yolo import TFLiteYOLO
from ModelInstances.yolo_tflite.prediction_processor_yolo import PredictionProcessorYOLO

model_path = default_config['model_paths']['tflite-yolo-v5']
label_map_path = default_config['label_maps']['COCO']

enet = TFLiteYOLO(model_path=model_path,
                          labels=default_config['label_maps']['COCO'],
                          prediction_processor=PredictionProcessorYOLO)

camera = CameraStreamDetect(model=enet)
camera.show_stream()
