from ModelInstances.yolo_tflite.yolo_model import YOLOTFLite
from Camera.camera_detect import CameraStreamDetect
from Config.default_config import default_config

model_path = default_config['model_paths']['tflite-yolo-v5']
label_map_path = default_config['label_maps']['COCO']

yolo_tf_lite = YOLOTFLite(model_path=model_path, label_map_path=label_map_path)

camera = CameraStreamDetect(model=yolo_tf_lite)
camera.show_stream()
