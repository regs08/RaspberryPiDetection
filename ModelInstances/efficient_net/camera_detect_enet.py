from Config.default_config import default_config
from ModelInstances.efficient_net.efficient_net_model import EfficientNetTFLite
from ModelInstances.efficient_net.prediction_processor_enet import PredictionProcessorENet
from Camera.camera_detect import CameraStreamDetect

model_path = default_config['model_paths']['efficient-net']

enet = EfficientNetTFLite(model_path=model_path,
                                  labels=default_config['label_maps']['imagenet'],
                                  prediction_processor=PredictionProcessorENet)

camera = CameraStreamDetect(model=enet)
camera.show_stream()