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

def match_template(image, template, method=cv2.TM_CCORR):
  result_image = cv2.matchTemplate(edged, template, cv2.TM_CCORR) 
  (_, max_val, _, max_loc) = cv2.minMaxLoc(result_image)
  return (max_val, max_loc)


print("Main: Full detector")
image_path = sys.argv[1]
image_name = image_path.split('/')[-1]
print(f"Opening {image_name}")
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Read the template
template = cv2.imread('templates/circle_dartboard.jpg', 0)
template = cv2.resize(template, (50, 50))
cv2.imwrite("templates/dart_circle.jpg", template)
template_height, template_width = template.shape

found = None
maximum_matches = []

lowest_scale = 0.2
highest_scale = 1.0
number_of_resizes = 40
for scale in np.linspace(lowest_scale, highest_scale, number_of_resizes)[::-1]:
  # resize the image according to the scale
  resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
  # resized = ip.resize_with_aspect_ratio(gray, max_side = int(gray.shape[1] * scale))
  # cv2.imwrite(f"preprocess/resized_{scale}_{image_name}", resized)

  # keep track of the ratio of the resizing
  ratio = gray.shape[1] / float(resized.shape[1]) 

  # if the resized image is smaller than the template, then break
  # from the loop 
  # detect edges in the resized, grayscale image and apply template
  # matching to find the template in the image 
  edged  = cv2.Canny(resized, 50, 200)
  # result = cv2.matchTemplate(edged, template, cv2.TM_CCORR) 
  # (_, max_val, _, maxLoc) = cv2.minMaxLoc(result) 
  max_val, max_loc = match_template(edged, template)
  # if we have found a new maximum correlation value, then update
  # the found variable 
  if found is None or max_val > found[0]: 
    print("Result:", max_val, max_loc)
    if resized.shape[0] < template_height or resized.shape[1] < template_width: 
      break
    found = (max_val, max_loc, ratio)
    maximum_matches.append((max_val, max_loc, ratio))

print("matches found:", len(maximum_matches))

for max_val, max_loc, ratio in maximum_matches:
  startX = int(max_loc[0] * ratio)
  startY = int(max_loc[1] * ratio)
  endX = int((max_loc[0] + template_width) * ratio)
  endY = int((max_loc[1] + template_height) * ratio)
  cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)

# draw a bounding box around the detected result and display the image 
# for top_left, bottom_right in detections:
#   cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

util.show_image(image)