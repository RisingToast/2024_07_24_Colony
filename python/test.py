import matplotlib.pyplot as plt
from ultralytics import YOLO

# 모델 로드
model = YOLO("yolov8n.pt")

# 모델 학습
model.train(data='C:/kkt/2024_07_24_Colony/roboflow_dataset/data.yaml', epochs = 100, patience = 300, batch = 32, imgsz = 500)

# 학습 모델 확인
print(type(model.names), len(model.names))
print(model.names)

# 이미지 예측
results = model.predict(source = 'C:/kkt/2024_07_24_Colony/IMG2/', save = True)

# 모델 저장
model.save('C:/kkt/2024_07_24_Colony/save_model/train_100_300_32_500.pt')


