import cv2
import numpy as np
import sys
import os
import math
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH); 
import util
import image_processing as ip

if __name__ == '__main__':
  image_path = sys.argv[1]
  image_name = image_path.split('/')[-1]
  print(f"Opening {image_name}")
  image = cv2.imread(image_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # max_side = 300
  # image = ip.resize_with_aspect_ratio(image, max_side=max_side)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  cascade_file = '../task2/dartcascade/cascade.xml'
  vj_detections = util.viola_jones_detection(cascade_file, gray)

  # threshold = 80
  # confirmed_detections = []
  for x, y, width, height in vj_detections:
    image = cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    # detection_region = CROP THE DETECTION REGION
    # htc = HougTransformCirle(detection_region)
    # htc.process_space(threshold=threshold)
    # circles = htc.detect_circles()
    # if (len((circles) >= 1):
      # htl = HougTransformLines()   # 4 -HT Lines
      # if (htl.number_of_lines()):
        # confirmed_detections.append(detection)
  
  # for x, y, width, height in confirmed_detections:
    # image = cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

  # Draw ground truth
  
  # Print Performance Report
  # cv2.imwrite(f'out/{image_name}', gradient)
  util.show_image(image)


  # gradient = ip.segment_by_threshold(magnitude, threshold=threshold)
  # cv2.imwrite(f'preprocess/grad_thre{image_name}', gradient)

  # grad_direction = ip.gradient_direction(gray)
  # cv2.imwrite(f'preprocess/dir_{image_name}', grad_direction)
  # 1 resize for efficiency



