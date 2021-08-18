import cv2
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt 

yolo = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

classes = []

with open("coco.names", 'r') as f:
    classes = f.read().splitlines()

#img = cv2.imread('1_traffic_light.jpg')
#img = cv2.imread('2_apple.jpg')
#img = cv2.imread('3_person.jpeg')
#img = cv2.imread('4_Img_Laptop.jpeg')    
img = cv2.imread('5_Stop_Sign.jpg')

height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), (0,0,0,), swapRB=True, crop = False )
# to print image
#i = blob[0].reshape(320,320,3)
#plt.imshow(i)  
yolo.setInput(blob)
output_layer_names = yolo.getUnconnectedOutLayersNames()
Layeroutput = yolo.forward(output_layer_names)

boxes = []
confidences = []
class_ids = []

for output in Layeroutput:
    for detection in output:
        score = detection[5:] # score is array form 1 - 80 
        class_id =np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.6:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2) # find corners
            y = int(center_y - h/2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
print(len(boxes))
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


font = cv2.FONT_HERSHEY_COMPLEX
colors = np.random.uniform(0,255, size = (len(boxes),3) )
if len(indexes) > 0:
    for i in indexes.flatten():
        x,y,w, h = boxes[i]
    
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
    
        cv2.rectangle (img, (x,y),(x+w, y+h), color, 1) 
        cv2.putText (img, label +" "+confidence, (x,y+20), font, 1, (255,255,255), 1)


cv2.imshow('IMAGE - OBJECT DETECTION', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


 


