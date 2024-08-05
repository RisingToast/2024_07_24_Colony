import os
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO

# 모델명
model = YOLO('C:/kkt/2024_07_24_Colony/save_model/train_50_300_16_512.pt')

# 이미지 예측
results = model.predict(source='C:/kkt/2024_07_24_Colony/IMG_ppt/add_ppt', save=True, max_det=5000)

# 결과 구조 확인 및 출력
for i, result in enumerate(results):
    count = len(result.boxes)  # 각 이미지에서 바운딩 박스의 수 세기
    print(f'{i + 1}번째 사진의 예측된 바운딩 박스의 개수: {count}')
    
    # 바운딩 박스 좌표 출력
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # 각 바운딩 박스의 좌표를 추출하여 정수형으로 변환
        print(f"Box coordinates: ({x1}, {y1}), ({x2}, {y2})")
    
    # 원본 이미지 로드 및 바운딩 박스 그리기
    image_path = result.path
    image = cv2.imread(image_path)
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 이미지 저장 (OpenCV BGR을 Matplotlib RGB로 변환)
    output_dir = 'C:/kkt/2024_07_24_Colony/predict_img_saved/Img_train_50_300_32_512'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, os.path.basename(image_path))
    plt.imsave(save_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

print("모든 이미지를 저장했습니다.")
