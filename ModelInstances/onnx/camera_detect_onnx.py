from Config.default_config import default_config
from ModelInstances.onnx.onnx_od import OnnxTFLite
from Camera.camera_detect import CameraStreamDetect

model_path = default_config['model_paths']['onnx-od-n']
classes = default_config['class_lists']['COCO']

onnx_tf_lite = OnnxTFLite(model_path=model_path, classes=classes)

camera = CameraStreamDetect(model=onnx_tf_lite)
camera.show_stream()