from ultralytics import YOLO

# Load a model

model = YOLO("C:/kkt/2024_07_24_Colony/save_model/train_100_300_16_512.pt")  # load a custom trained model

# Export the model
model.export(format="tflite")