from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-pose.pt')  # load an official model
model = YOLO('runs/pose/train/weights/best.pt')  # load a custom model

# Predict with the model
results = model('dataset/tests/armor_1_test.mp4', save = True, device = 0,hide_conf = True, hide_labels = True)  # predict on an image