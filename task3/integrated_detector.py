import cv2
import numpy as np
import sys
import os
import math
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH); 
import util
import image_processing as ip
from hough_transform_circles import HoughTransformCircles

if __name__ == '__main__':
  image_path = sys.argv[1]
  image_name = image_path.split('/')[-1]
  print(f"Opening {image_name}")
  image = cv2.imread(image_path)

  # max_side = 300
  # image = ip.resize_with_aspect_ratio(image, max_side=max_side)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  images_to_dartboards = util.load_ground_truth("task2/dartboards.json")
  ground_truth_dartboards = images_to_dartboards[image_name]
  number_of_dartboards = len(ground_truth_dartboards)
  print("Number of dartboards:", number_of_dartboards)

  for x1, y1, x2, y2 in ground_truth_dartboards:
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

  cascade_file = 'task2/dartcascade/cascade.xml'
  vj_detections = util.viola_jones_detection(cascade_file, gray)
  for x, y, width, height in vj_detections:
    image = cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)

  threshold = 50
  confirmed_detections = []
  for x, y, width, height in vj_detections:
    detection_index = 0
    detection_region = gray[y : y+height, x : x+width]
    cv2.imwrite(f'task3/preprocess/dr_{detection_index}.jpg', detection_region)
    htc = HoughTransformCircles(detection_region)
    htc.process_space(threshold=threshold)
    circles = htc.detect_circles(minimum_votes=15)
    print("Detected Circles", len(circles))
    if (len(circles) >= 1):
      confirmed_detections.append((x, y, width, height))

  for x, y, width, height in confirmed_detections:
    image = cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

  util.print_report(ground_truth_dartboards, confirmed_detections)
  
  util.show_image(image)

  # gradient = ip.segment_by_threshold(magnitude, threshold=threshold)
  # cv2.imwrite(f'preprocess/grad_thre{image_name}', gradient)

  # grad_direction = ip.gradient_direction(gray)
  # cv2.imwrite(f'preprocess/dir_{image_name}', grad_direction)
  # 1 resize for efficiency



