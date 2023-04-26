from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-pose.pt')  # load an official model
model = YOLO('runs/pose/train/weights/best.pt')  # load a custom model

# Predict with the model
results = model('datasets/armor-four-points/tests/blue_1.jpg', save = True, device = 0)  # predict on an image