from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-pose.pt')  # load an official model
model = YOLO('runs/pose/train6/weights/best.pt')  # load a custom model

# Predict with the model
results = model('dataset/images/train/280.jpg', save = True, device = 0)  # predict on an image