import mss
import cv2
import numpy as np
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5/runs/train/exp3/weights/best.pt', force_reload=True)
model.conf = 0.7 #信心指數下限
def capture_screen():
    with mss.mss() as sct:
        # 設定截圖區域，這裡設定為整個螢幕
        monitor = sct.monitors[1]  # 若只有一個螢幕，則使用索引0
        left = monitor["left"]
        top = monitor["top"]
        width = monitor["width"]
        height = monitor["height"]
        monitor_area = (left, top, left + width, top + height)

        while True:
            # 截圖
            img = sct.grab(monitor_area)

            # 將圖像轉換為OpenCV格式
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

            results = model(img)

            # 顯示截取的畫面
            # cv2.imshow('Screen Capture', img)
            cv2.imshow('YOLO COCO 03 mask detection', np.squeeze(results.render()))

            # 按下q鍵退出迴圈
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 關閉視窗
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_screen()
