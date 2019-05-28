# -*- coding: UTF-8 -*-

import os, cv2
from PIL import Image, ImageDraw

# 检测人脸
def detect(path, xml):
    detector = cv2.CascadeClassifier(xml)
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.15, 5)

    # if len(faces) > 0:
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(image, (x, y), (x + w, y + w), (255, 255, 0), 2)
    #
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return faces

# 保存截图
def saveCrop(path, faces, destination):
    imageName = os.path.basename(path).split('.')
    count = 0
    for (x, y, w, h) in faces:
        savePath = os.path.join(destination, imageName[0] + str(count) + "." + imageName[1])
        Image.open(path).crop((x, y, x + w, y + w)).save(savePath)
        count += 1

# 保存
def saveDraw(path, faces, destination):
    image = Image.open(path)
    imageName = os.path.basename(path).split('.')
    draw = ImageDraw.Draw(image)
    for (x, y, w, h) in faces:
        draw.rectangle((x, y, x + w, y + w), outline = (255, 255, 0))

    savePath = os.path.join(destination, imageName[0] + "_draw." + imageName[1])
    image.save(savePath)

if __name__ == '__main__':
    src = "images"
    destination = "result"
    if not os.path.exists(destination):
        os.makedirs(destination)

    files = os.listdir(src)
    for file in files:
        if not os.path.isdir(file):
            path = os.path.join(src, file)
            faces = detect(path, "haarcascades/haarcascade_frontalface_alt.xml")
            saveCrop(path, faces, destination)
            saveDraw(path, faces, destination)



