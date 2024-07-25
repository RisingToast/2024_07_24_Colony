from ultralytics import YOLO

model = YOLO("yolov8s.pt") # load the model
results = model.train(data="coco128.yaml", epochs=100)
results = model("./image.png")