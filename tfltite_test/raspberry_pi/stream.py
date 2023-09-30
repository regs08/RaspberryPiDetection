from flask import Flask, Response
from object_detector import ObjectDetector
model_path = "/Users/cole/PycharmProjects/TFLite/tfltite_test/raspberry_pi/efficientdet_lite0.tflite"
app = Flask(__name__)
detector = ObjectDetector(model_path)

@app.route('/video')
def video():
    return Response(detector.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
