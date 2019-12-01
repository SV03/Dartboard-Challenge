import cv2
import numpy as np
import sys
import os
import math
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH); 
import util
import image_processing as ip
import imutils

print("Main: Full detector")
image_path = sys.argv[1]
image_name = image_path.split('/')[-1]
print(f"Opening {image_name}")
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Read the template 
template = cv2.imread('../input/dart16.jpg', 0)
template = cv2.resize(template, (60, 50))
template_height, template_width = template.shape

found = None
detections = []

for scale in np.linspace(0.2, 1.0, 40)[::-1]:
  print(scale)
  
  # resize the image according to the scale, and keep track  
  # of the ratio of the resizing 
  
  resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
  # resized = ip.resize_with_aspect_ratio(gray, max_side = int(gray.shape[1] * scale))

  # cv2.imwrite(f"preprocess/resized_{scale}_{image_name}", resized)
  ratio = gray.shape[1] / float(resized.shape[1]) 
  
  # if the resized image is smaller than the template, then break 
  # from the loop 
  # detect edges in the resized, grayscale image and apply template  
  # matching to find the template in the image 
  edged  = cv2.Canny(resized, 50, 200) 
  result = cv2.matchTemplate(edged, template,cv2.TM_CCORR) 
  (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result) 
  # print("Result",( maxVal, maxLoc))
  # if we have found a new maximum correlation value, then update
  # the found variable 
  if found is None or maxVal > found[0]: 
    if resized.shape[0] < template_height or resized.shape[1] < template_width: 
      break
    found = (maxVal, maxLoc, ratio)
  
# unpack the found varaible and compute the (x, y) coordinates 
# of the bounding box based on the resized ratio 
    # (_, maxLoc, ratio) = found 
    startX = int(maxLoc[0] * ratio)
    startY = int(maxLoc[1] * ratio)
    endX = int((maxLoc[0] + template_width) * ratio)
    endY = int((maxLoc[1] + template_height) * ratio)
    detections.append(((startX, startY), (endX, endY)))

# draw a bounding box around the detected result and display the image 

for top_left, bottom_right in detections:
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2) 

util.show_image(image)