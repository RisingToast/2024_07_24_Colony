# 2024_07_24_Colony
원본 페이지 : [https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/]
> 밑의 내용은 원본 페이지의 한국어 번역본입니다. <br>
번역기를 사용했기 때문에 정확성이 떨어질 수 있습니다.

# 사용자 지정 데이터 훈련하기

이 가이드에서는 YOLOv5 🚀를 사용하여 사용자 지정 데이터 세트를 훈련하는 방법을 설명합니다.

시작하기 전
PyTorch>=1.8을 포함한 Python>=3.8.0 환경에서 저장소를 복제하고 요구사항.txt를 설치합니다. 모델과 데이터 세트는 최신 YOLOv5 릴리스에서 자동으로 다운로드됩니다.

``` 
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```
# 사용자 지정 데이터에 대한 교육

객체를 감지하는 사용자 지정 모델을 만드는 것은 이미지를 수집 및 정리하고, 관심 객체에 레이블을 지정하고, 모델을 학습시키고, 야생에 배포하여 예측을 수행한 다음, 배포된 모델을 사용하여 엣지 사례의 예를 수집하여 반복 및 개선하는 반복적인 프로세스입니다.<br>

라이선스<br>
Ultralytics는 두 가지 라이선스 옵션을 제공합니다:<br>
- 학생과 애호가에게 이상적인 오픈 소스 라이선스인 AGPL-3.0 라이선스.<br>

- 엔터프라이즈 라이선스 - 제품 및 서비스에 당사의 AI 모델을 통합하고자 하는 기업을 위한 라이선스입니다.<br>
자세한 내용은 울트라 애널리틱스 라이선스를 참조하세요.<br>

해당 데이터에서 객체의 클래스를 학습하려면 YOLOv5 모델을 레이블이 지정된 데이터로 학습해야 합니다. 학습을 시작하기 전에 데이터 세트를 만드는 데는 두 가지 옵션이 있습니다:

# 옵션 1: 로보플로우 데이터 세트 만들기

## 1.1 이미지 수집

모델은 모범을 통해 학습합니다. 

실제와 유사한 이미지로 훈련하는 것이 가장 중요합니다.<br> 최종적으로 프로젝트를 배포할 때와 동일한 구성(카메라, 각도, 조명 등)으로 다양한 이미지를 수집하는 것이 가장 이상적입니다.<br>

이것이 불가능하다면 공개 데이터 세트에서 시작하여 초기 모델을 훈련한 다음 추론 중에 야생에서 이미지를 샘플링하여 데이터 세트와 모델을 반복적으로 개선할 수 있습니다.

## 1.2 레이블 만들기

이미지를 수집한 후에는 관심 있는 객체에 주석을 달아 모델이 학습할 근거 자료를 만들어야 합니다.<br>
Roboflow Annotate는 팀과 함께 이미지를 관리하고 라벨을 붙이고 YOLOv5의 주석 형식으로 내보내기 위한 간단한 웹 기반 툴입니다.

## 1.3 YOLOv5용 데이터 세트 준비하기

이미지에 Roboflow로 레이블을 지정하든 지정하지 않든, 데이터 세트를 YOLO 형식으로 변환하고, YOLOv5 YAML 구성 파일을 생성하고, 교육 스크립트로 가져올 수 있도록 호스팅하는 데 사용할 수 있습니다.<br>
무료 Roboflow 계정을 만들고 공개 워크스페이스에 데이터 세트를 업로드한 다음, 주석이 없는 이미지에 라벨을 붙인 다음, 데이터 세트의 버전을 생성하고 내보내 YOLOv5 Pytorch 형식으로 만듭니다.<br>
참고: YOLOv5는 훈련 중에 온라인 증강을 수행하므로 YOLOv5를 사용한 훈련에는 Roboflow에서 어떤 증강 단계도 적용하지 않는 것이 좋습니다. 하지만 다음과 같은 전처리 단계를 적용하는 것이 좋습니다:<br>

**자동 방향** - 이미지에서 EXIF 방향을 제거합니다.<br>
**크기 조정(늘이기)** - 모델의 정사각형 입력 크기로 조정합니다(640x640이 YOLOv5 기본값).
버전을 생성하면 데이터 세트의 스냅샷이 생성되므로 나중에 이미지를 더 추가하거나 구성을 변경하더라도 언제든지 돌아가서 향후 모델 학습 실행을 이 버전과 비교할 수 있습니다.

버전을 생성하면 데이터 세트의 스냅샷이 생성되므로 나중에 이미지를 더 추가하거나 구성을 변경하더라도 언제든지 돌아가서 향후 모델 학습 실행을 이 버전과 비교할 수 있습니다.

YOLOv5 파이토치 형식으로 내보낸 다음 스니펫을 교육 스크립트나 노트북에 복사하여 데이터 세트를 다운로드합니다.

# 옵션 2: 수동 데이터 세트 만들기

## 2.1 데이터세트.yaml 만들기

COCO128은 COCO train2017의 첫 128개 이미지로 구성된 작은 튜토리얼 데이터 세트의 예시입니다. 

이 128개의 이미지는 훈련 파이프라인이 오버피팅이 가능한지 확인하기 위해 훈련과 검증 모두에 사용됩니다. <br>
아래에 표시된 data/coco128.yaml은 
1) 데이터 세트 루트 디렉토리 경로와 훈련/val/테스트 이미지 디렉터리(또는 이미지 경로가 포함된 *.txt 파일)의 상대 경로, 
2) 클래스 이름 사전을 정의하는 데이터 세트 구성 파일입니다:
```python 
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco128 # dataset root dir
train: images/train2017 # train images (relative to 'path') 128 images
val: images/train2017 # val images (relative to 'path') 128 images
test: # test images (optional)

# Classes (80 COCO classes)
names:
    0: person
    1: bicycle
    2: car
    # ...
    77: teddy bear
    78: hair drier
    79: toothbrush
```

## 2.2 레이블 만들기
주석 도구를 사용하여 이미지에 라벨을 붙인 후 이미지당 하나의 *.txt 파일을 사용하여 라벨을 YOLO 형식으로 내보냅니다(이미지에 객체가 없는 경우 *.txt 파일이 필요하지 않음). .txt 파일 사양은 다음과 같습니다:
- 개체당 한 행
- 각 행은 클래스 x_center y_center 너비 높이 형식입니다.
- 상자 좌표는 정규화된 xywh 형식(0에서 1 사이)이어야 합니다. 상자가 픽셀 단위인 경우 x_center와 너비를 이미지 너비로 나누고, y_center와 높이를 이미지 높이로 나눕니다.
- 클래스 번호는 영 인덱싱됩니다(0부터 시작).

## 2.3 디렉토리 정리

아래 예시에 따라 기차 및 밸 이미지와 레이블을 구성합니다. YOLOv5는 /coco128이 /yolov5 디렉토리 옆의 /datasets 디렉토리 안에 있다고 가정합니다. YOLOv5는 각 이미지 경로에서 /이미지/의 마지막 인스턴스를 /라벨/로 대체하여 각 이미지에 대한 라벨을 자동으로 찾습니다. 예를 들어
``` 
../datasets/coco128/images/im0.jpg  # image
../datasets/coco128/labels/im0.txt  # label
```

# 3. 모델 선택
훈련을 시작할 사전 훈련된 모델을 선택합니다. 여기서는 두 번째로 작고 빠른 모델인 YOLOv5s를 선택합니다. 모든 모델에 대한 전체 비교는 README 표를 참조하세요.<br>
[https://github.com/ultralytics/assets/releases/download/v0.0.0/model_comparison.png]

# 4. 훈련
데이터 세트, 배치 크기, 이미지 크기를 지정하고 사전 학습된 --weights yolov5s.pt(권장) 또는 무작위로 초기화된 --weights '' --cfg yolov5s.yaml(권장하지 않음)을 사용하여 COCO128에서 YOLOv5s 모델을 학습합니다. 사전 학습된 가중치는 최신 YOLOv5 릴리스에서 자동으로 다운로드됩니다.
```
python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt
```
모든 훈련 결과는 실행 디렉터리가 증가함에 따라 runs/train/에 저장됩니다(예: runs/train/exp2, runs/train/exp3 등). 자세한 내용은 튜토리얼 노트북의 트레이닝 섹션을 참조하세요.

# 5. 시각화

### 혜성 로깅 및 시각화 🌟 새로운 기능
이제 Comet이 YOLOv5와 완전히 통합되었습니다. 실시간으로 모델 메트릭을 추적 및 시각화하고, 하이퍼파라미터, 데이터 세트 및 모델 체크포인트를 저장하고, Comet 사용자 지정 패널로 모델 예측을 시각화할 수 있습니다! Comet을 사용하면 작업 내용을 놓치지 않고 모든 규모의 팀에서 결과를 쉽게 공유하고 협업할 수 있습니다!

시작은 간단합니다:
```
pip install comet_ml  # 1. install
export COMET_API_KEY=<Your API Key>  # 2. paste API key
python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt  # 3. train
```
이 연동에서 지원되는 모든 Comet 기능에 대해 자세히 알아보려면 Comet 튜토리얼을 확인하세요. Comet에 대해 자세히 알아보고 싶으시다면 설명서를 참조하세요. 먼저 Comet Colab Notebook을 사용해 보세요: 