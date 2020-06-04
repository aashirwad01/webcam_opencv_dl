import cv2
import numpy as np
import time
import sys
sys.path.insert(0, '/home/aashirwad/Desktop/python_virtual/darknet/')
print(sys.path)

# Load Yolo
net = cv2.dnn.readNet("/home/aashirwad/Desktop/python_virtual/darknet/yolov3-tiny.weights", "/home/aashirwad/Desktop/python_virtual/darknet/cfg/yolov3-tiny.cfg")
classes = []
with open("/home/aashirwad/Desktop/python_virtual/darknet/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#colors = np.random.uniform(0, 255, size=(len(classes), 3))
def dis(first,second):
  return(((center_co[i][0]-center_co[j][0])**2)+((center_co[i][1]-center_co[j][1])**2))**1/2
url = "http://192.168.43.6:8080" 
cap = cv2.VideoCapture(url+"/video")
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape
     # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    center_co=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                center_co.append([center_x,center_y])
                
				
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i in indexes and j in indexes and class_ids[i]==0 and class_ids[j]==0:
                if (i!=j):
                    y=dis(i,j)
                    print(y)
                    if (y <5000000):
                        cv2.line(frame,(center_co[i][0],center_co[i][1]), (center_co[j][0],center_co[j][1]), (0,0,255), 2)
                        x, y, w, h = boxes[i]
                        x1,y1,w1,h1=boxes[j]
                        label = str(classes[class_ids[i]])
                        
                        #confidence = confidences[i]
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
                        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), color, 2)
                        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + 30), color, -1)
                        #cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)
                        
            
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
    cv2.imshow("ff",frame)
    cv2.waitKey(1)






    #for i in range(len(boxes)):
       # if i in indexes:
            #x, y, w, h = boxes[i]
            #label = str(classes[class_ids[i]])
            #confidence = confidences[i]
            #color = (255, 0, 0)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            #cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
            #cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)
            #elapsed_time = time.time() - starting_time
            #fps = frame_id / elapsed_time
            #cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
            #cv2.imshow("ff",frame)
            #key = cv2.waitKey(1)
            #if key == 27:
             # break

cap.release()
cv2.destroyAllWindows()
