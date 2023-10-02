import cv2
import math
from pathlib import Path
from yolov8.models.experimental import attempt_load
from yolov8.utils.datasets import LoadStreams
from yolov8.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov8.utils.plots import Annotator, colors
from yolov8.utils.torch_utils import select_device
import torch

# Set your YOLO model weights path
weights_path = 'path/to/your/yolov8n.pt'

# Set your class names
classNames = ["kid", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Initialize the YOLO model
device = select_device()
model = attempt_load(weights_path, map_location=device)
imgsz = check_img_size(640, s=model.stride.max())

# Set up the webcam or video capture
video_path = '/path/to/your/video/KIDS3.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

while True:
    ret, img0 = cap.read()
    if not ret:
        break

    img = img0.copy()

    # Preprocess the image
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Get detections
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    annotator = Annotator(imgsz, line_thickness=3)
    if pred[0] is not None:
        detections = pred[0]
        detections = scale_coords(img.shape[2:], detections[:, :4], img0.shape).round()

        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f'{classNames[int(cls)]} {conf:.2f}'
            color = colors(int(cls))
            cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
            annotator.text((x1, y1), label, color)

    cv2.imshow('YOLO', img0)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()