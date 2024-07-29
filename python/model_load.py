import matplotlib.pyplot as plt
from ultralytics import YOLO

# 학습된 모델 로드
model = YOLO('C:/kkt/2024_07_24_Colony/save_model/train_50_300_16_500.pt')

# 이미지 예측
results = model.predict(source='C:/kkt/2024_07_24_Colony/IMG2/', save=True)

# 예측 결과 확인
print(results)

# 바운딩 박스 개수 세기 및 이미지 출력
for result in results:
    count = len(result.boxes)  # 각 이미지에서 바운딩 박스의 수 세기
    print(f'예측된 바운딩 박스의 개수: {count}')
    
    # 예측된 바운딩 박스가 포함된 이미지 보기
    ##result.show()  # 결과를 이미지로 띄움
