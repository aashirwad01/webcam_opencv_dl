import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

sys.path.insert(0, '/home/aashirwad/Desktop/python_virtual/mask_rcnn/')
print(sys.path)


from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

sys.path.insert(0, '/home/aashirwad/Desktop/python_virtual/mask_rcnn/samples/coco')
print(sys.path)

import coco

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

# Load weights trained on MS-COCO
model.load_weights('/home/aashirwad/Desktop/python_virtual/mask_rcnn/samples/mask_rcnn_coco.h5', by_name=True)

# COCO Class names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane','bus', 'train', 'truck', 'boat', 'traffic light','fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
               
def dis(first,second):
	return((((((r['rois'][first][1]+r['rois'][first][3])/2)-((r['rois'][second][1]+r['rois'][second][3])/2))**2)+((((r['rois'][first][0]+r['rois'][first][2])/2)-((r['rois'][second][0]+r['rois'][second][2])/2))**2))**1/2) 
# capture the video

url = "http://192.168.43.6:8080" 
cap = cv2.VideoCapture(url+"/video")

# check if capture was successful
if not cap.isOpened(): 
    print("Could not open!")
else:
    print("Video read successful!")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print('Total frames: ' + str(total_frames))
    print('width: ' + str(width))
    print('height: ' + str(height))
    print('fps: ' + str(fps)) 
cap.release()
fps=3   	   
count=0
#pathut='vide.mp4'

size = (width,height)
#out = cv2.VideoWriter(pathut,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
url = "http://192.168.43.6:8080" 
cap = cv2.VideoCapture(url+"/video")
while cap.isOpened():
    
    ret, image = cap.read()
    
    
    if not ret:
      break
    else:
      if (count%8==0):
        
        
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
  
        newimage = image
        for i in range(len(r['rois'])):
          for j in range(len(r['rois'])):
            if(r['class_ids'][i]==1 and r['class_ids'][j]==1):
        
              start_point = (math.floor((r['rois'][i][1]+r['rois'][i][3])/2),math.floor((r['rois'][i][0]+r['rois'][i][2])/2))
        
   
              end_point = (math.floor((r['rois'][j][1]+r['rois'][j][3])/2),math.floor((r['rois'][j][0]+r['rois'][j][2])/2))
        
    
      
              y=dis(i,j)
        
              if (y<5000):
          
                color = (255, 0, 0)
                thickness = 2
                if (i!=j):
                  image_dis= cv2.line(newimage, start_point, end_point, color, thickness)

        #out.write(newimage)
        cv2.imshow('n',newimage)
	
        #print(count)
    
    count=count+1
#out.release()
  
print('Done')
cap.release()
cv2.destroyAllWindows()

