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
import non_maximum_suppression as nms

def read_templates(names_list):
  templates = []
  for template_name in names_list:
    templates.append(cv2.imread(template_name, 0))
  return templates

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

template_names = [
  "templates/dart_circle.jpg",
  # "templates/dart_ellipsis0.jpg",
  # "templates/dart_ellipsis1.jpg",
  # "templates/dart_ellipsis2.jpg",
  # "templates/dart_ellipsis3.jpg", 
]
templates = read_templates(template_names)

max_found = None
maximum_matches = []
points_nms = []
lowest_scale = 0.2
highest_scale = 1.0
number_of_resizes = 40
for scale in np.linspace(lowest_scale, highest_scale, number_of_resizes)[::-1]:
  # resize the image according to the scale
  resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
  # resized = ip.resize_with_aspect_ratio(gray, max_side=int(gray.shape[1] * scale))
  # cv2.imwrite(f"preprocess/resized_{scale}_{image_name}", resized)

  # keep track of the ratio of the resizing
  ratio = gray.shape[1] / float(resized.shape[1]) 

  edged  = cv2.Canny(resized, 50, 200)
  for template in templates:
    template_height, template_width = template.shape
    # if the resized image is smaller than the template, then break from the loop
    if resized.shape[0] < template_height or resized.shape[1] < template_width: 
      break

    max_val, max_loc = match_template(edged, template)
  # if a new max_val is found we store it and update the max_found variable
    if max_found is None or max_val > max_found[0]: 
      max_found = (max_val, max_loc, ratio)
      startX = int(max_loc[0] * ratio)
      startY = int(max_loc[1] * ratio)
      endX = int((max_loc[0] + template_width) * ratio)
      endY = int((max_loc[1] + template_height) * ratio)
      maximum_matches.append((max_val, (startX, startY), (endX, endY)))
      points_nms.append((startX, startY, endX, endY))
      print("New max:", max_val, (startX, startY), (endX, endY))


boxes = nms.non_max_suppression_fast(np.array(points_nms), 0.6)
print(boxes)
print("Matches Found:", len(maximum_matches))

for max_val, top_left, bottom_right in maximum_matches:
  cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 3)

for x1, y1, x2, y2 in boxes:
  cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# for top_left, bottom_right in detections:
#   cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

util.show_image(image)