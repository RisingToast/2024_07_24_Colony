from ultralytics import YOLO

# 학습된 모델 로드
model = YOLO('C:/kkt/2024_07_24_Colony/save_model/train_10_300_64_500.pt')

# 이미지 예측
results = model.predict(source='C:/kkt/2024_07_24_Colony/IMG2/51.cropped-aureus-colony-counting.jpg', save=True)

# 예측 결과 확인
print(results)
