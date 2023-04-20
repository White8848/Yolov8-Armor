from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8m-pose.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n-pose.yaml').load('runs/pose/train/weights/best.pt')  # build from YAML and transfer weights

if __name__ == '__main__':
    # Train the model, device = 0 for GPU, device = 'cpu' for CPU
    model.train(data='armor-four-points.yaml', epochs=100, imgsz=640, device = 0)
