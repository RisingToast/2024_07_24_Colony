import matplotlib.pyplot as plt
from ultralytics import YOLO

# 모델 로드
model = YOLO("D:/git_colony/2024_07_24_Colony/save_model/train_50_300_32_512.pt")

# 학습 모델 확인
print(type(model.names), len(model.names))
print(model.names)

# 이미지 예측
results = model.predict(source = 'D:/git_colony/2024_07_24_Colony/IMG_ppt', save = True)