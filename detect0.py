import torch
import cv2
import numpy as np

# model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5/runs/train/exp6/weights/best.pt', force_reload=True)
model.conf = 0.7 #信心指數下限
# print(model)
img=cv2.imread('images/0.png')
results = model(img)
results.print()
print(results.xyxy)
cv2.namedWindow('YOLO COCO', 0)
cv2.resizeWindow('YOLO COCO', 800, 480)
cv2.imshow('YOLO COCO', np.squeeze(results.render()))
cv2.waitKey(0)
