import cv2
import os
import numpy as np
from keras.models import model_from_json

json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model/model.h5")
print("Loaded model from disk")

prototxtPath = 'Face Detection Cafe Model/deploy.prototxt'
weightsPath = 'Face Detection Cafe Model/res10_300x300_ssd_iter_140000.caffemodel'

net = cv2.dnn.readNet(prototxtPath,weightsPath)

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()
    
    (h,w) = img.shape[:2]
    
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
	(104.0, 177.0, 123.0))
    
    
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.2:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            cv2.rectangle(img, (startX, startY), (endX,endY), (0, 0, 255), 4)
            org = (startX,endY+25)
            # font 
            font = cv2.FONT_HERSHEY_SIMPLEX
            # fontScale 
            fontScale = 1
            
            # Blue color in BGR 
            color = (255, 255, 255)
            
            # Line thickness of 2 px 
            thickness = 2
            
            img_crp = img[startY:endY,startX:endX]
            img_crp = cv2.cvtColor(img_crp,cv2.COLOR_BGR2GRAY)
            img_crp = cv2.resize(img_crp,(50,50))
        
            img_crp = img_crp.reshape(-1,50,50,1)
            img_crp = img_crp/255
            ##print(img_crp.shape)
            pred = loaded_model.predict(img_crp.reshape(-1,50,50,1))
            my_list = map(lambda x: x[0], pred)
            pred = list(my_list)[0]
            
            
            if pred > 0.3:
                cv2.rectangle(img,(startX,endY),(endX,endY+30),(0,0,255),-1)
                cv2.putText(img, 'Mask', org, font,fontScale, color, thickness, cv2.LINE_AA) 
            else:
                cv2.rectangle(img,(startX,endY),(endX,endY+30),(0,0,255),-1)
                cv2.putText(img, 'No Mask', org, font,fontScale, color, thickness, cv2.LINE_AA) 
        
        
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()        