import matplotlib.pyplot as plt
from ultralytics import YOLO

# 모델 로드
model = YOLO("yolov8n.pt")

# 모델 학습
model.train(data='C:/kkt/2024_07_24_Colony/roboflow_dataset/data.yaml', epochs = 300, patience = 300, batch = 64, imgsz = 100)

# 학습 모델 확인
print(type(model.names), len(model.names))
print(model.names)

# 이미지 예측
results = model.predict(source = 'C:/kkt/2024_07_24_Colony/IMG2/', save = True)


