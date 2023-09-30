import pytesseract
from PIL import Image


image = Image.open('images/lala.png')

result1 = pytesseract.image_to_string(image,lang='chi_tra')
print("识别结果：", result1)