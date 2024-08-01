import cv2
import os
from matplotlib import pyplot as plt
from ultralytics import YOLO

# 학습된 모델 로드
model = YOLO('C:/kkt/2024_07_24_Colony/runs/detect/train14_50_300_16/weights/best.pt')

# 이미지 예측
results = model.predict(source='C:/kkt/2024_07_24_Colony/IMG2/', save=True)

# 예측 결과 확인
print(results)

# 바운딩 박스 개수 세기 및 이미지 저장
output_dir = 'C:/kkt/2024_07_24_Colony/predict_img_saved/Img_train_50_16_best'
os.makedirs(output_dir, exist_ok=True)

counter = 1

for result in results:
    count = len(result.boxes)  # 각 이미지에서 바운딩 박스의 수 세기
    print(f'{counter}번째 사진의 예측된 바운딩 박스의 개수: {count}')
    counter += 1
    
    # 원본 이미지 로드
    image_path = result.path
    image = cv2.imread(image_path)
    
    # 바운딩 박스 그리기
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # 각 바운딩 박스의 좌표를 추출하여 정수형으로 변환
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 이미지 저장 (OpenCV BGR을 Matplotlib RGB로 변환)
    save_path = os.path.join(output_dir, os.path.basename(image_path))
    plt.imsave(save_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
print("모든 이미지를 저장했습니다.")
