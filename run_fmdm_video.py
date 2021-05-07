import sys
import cv2
import cvlib as cv
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 


model = load_model('./models/face_mask_detector.h5')
imsize = 224

vidcap = cv2.VideoCapture(sys.argv[1])

width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 25.0, (int(width), int(height)) )

while (True):
    ret, frame = vidcap.read()
    
    if not ret:
        break

    faces, _ = cv.detect_face(frame, threshold=0.5)

    for face in faces:

        startX, startY, endX, endY = face
        h, w = frame.shape[:2]

        if endX > w or endY > h:
            continue
        input_img = frame[startY:endY, startX:endX]
        input_img = cv2.resize(input_img, (imsize, imsize),
                            interpolation=cv2.INTER_AREA)
        input_img = img_to_array(input_img)
        input_img = np.expand_dims(input_img, axis=0)
        input_img = preprocess_input(input_img)

        pred = model.predict(input_img)
        if pred[0, 0] < 0.5:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            # print(f"{pred[0, 1]*100:6.2f} %")
        else:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # print(f"{pred[0, 0]*100:6.2f} %")  
        out.write(frame)

out.releas()
cv2.destroyAllWindows()