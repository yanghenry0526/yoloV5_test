from PIL import ImageGrab
import time

while True:
    img = ImageGrab.grab()
    img.show()
    time.sleep(0)