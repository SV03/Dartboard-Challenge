import cv2
import numpy as np
import sys
import os
import math
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH); 
import util
import image_processing as ip
from matplotlib import pyplot as plt
import imutils
   
# Read the main image 
img_rgb = cv2.imread('../input/dart14.jpg') 
   
# Convert to grayscale 
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 
   
# Read the template 
template = cv2.imread('../input/dart16.jpg',0) 
template = cv2.resize(template, (60, 50))
# Store width and height of template in w and h 
w, h = template.shape[::-1] 
found = None
  
for scale in np.linspace(0.2, 1.0, 40)[::-1]: 
  
    # resize the image according to the scale, and keep track  
    # of the ratio of the resizing 
    resized = imutils.resize(img_gray, width = int(img_gray.shape[1] * scale)) 
    r = img_gray.shape[1] / float(resized.shape[1]) 
   
    # if the resized image is smaller than the template, then break 
    # from the loop 
    # detect edges in the resized, grayscale image and apply template  
    # matching to find the template in the image 
    edged  = cv2.Canny(resized, 50, 200) 
    result = cv2.matchTemplate(edged, template,cv2.TM_CCORR) 
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result) 
    print("Result",( maxVal, maxLoc))
    # if we have found a new maximum correlation value, then update 
    # the found variable 
#     threshold = 0.4
#     loc = np.where( result >= threshold)
    if found is None or maxVal > found[0]: 
        if resized.shape[0] < h or resized.shape[1] < w: 
                break
        found = (maxVal, maxLoc, r) 
   
# unpack the found varaible and compute the (x, y) coordinates 
# of the bounding box based on the resized ratio 
        (_, maxLoc, r) = found 
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r)) 
        (endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r)) 
  
# draw a bounding box around the detected result and display the image 
        cv2.rectangle(img_rgb, (startX, startY), (endX, endY), (0, 255, 0), 2) 
util.show_image(img_rgb)