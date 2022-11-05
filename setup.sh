git clone https://github.com/WongKinYiu/yolov7.git
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install opencv-python install -qr requirements.txt
pip install mediapipe-silicon
pip uninstall protobuf
pip install protobuf==3.20.1