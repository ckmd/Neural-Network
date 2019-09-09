import cv2, os, glob, re
import numpy as np

def splitstring(word):
    xpos, xneg, ypos, yneg = 0,0,0,0
    if(word[12] == "0"):# & word[16] == "0"):
        xpos = 0
    else:
        if(word[11] == "-"):
            xneg = int(word[12:14])
        else:
            xpos = int(word[12:14])
    if(word[-2:] == "+0"):
        ypos = 0
    else:
        if(word[-3] == "-"):
            yneg = int(word[-2:])
        else:
            ypos = int(word[-2:])
    # final label (yaw+, yaw-, pitch+, pitch-)
    return xpos, xneg, ypos, yneg

# read several image
img_dir = "face" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*jpg')
files = glob.glob(data_path)
detected = 0
data = []
label = []
for f1 in files:
    image = cv2.imread(f1)
    # getting value of yaw and pitch
    base = os.path.basename(f1)
    base = os.path.splitext(base)
    pose = splitstring(base[0])

    # read patern
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    if(len(faces) == 0):
        continue # continue if there is no face
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h , x:x+w]

    # Gaussian Image Pyramid
    layer = roi_gray.copy()
    gaussian_pyramid = [layer]
    for i in range(1):
        layer = cv2.pyrDown(layer)
        gaussian_pyramid.append(layer)

    # cv2.imshow('layer1', gray)
    # cv2.waitKey(1000)

    detected += 1
    # print("face ", detected,"pose (x+, x-, y+, y-)", pose)
    data.append(layer)
    label.append(pose)

data = np.array(data)
label = np.array(label)/100
print(label)
# print("detected face : ", detected)
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)