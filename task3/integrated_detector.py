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
from hough_transform_lines import HoughTransformLines

if __name__ == '__main__':
  image_path = sys.argv[1]
  image_name = image_path.split('/')[-1]
  print(f"Opening {image_name}")
  image = cv2.imread(image_path)

  # max_side = 300
  # image = ip.resize_with_aspect_ratio(image, max_side=max_side)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  images_to_dartboards = util.load_ground_truth("../task2/dartboards.json")
  ground_truth_dartboards = images_to_dartboards[image_name]
  number_of_dartboards = len(ground_truth_dartboards)
  print("Number of dartboards:", number_of_dartboards)

  for x1, y1, x2, y2 in ground_truth_dartboards:
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

  cascade_file = '../task2/dartcascade/cascade.xml'
  vj_detections = util.viola_jones_detection(cascade_file, gray)
  for x, y, width, height in vj_detections:
    image = cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)

  threshold = 50
  magnitude = ip.extract_edges(gray, "GRAD")
  thresholded_magnitude = ip.segment_by_threshold(magnitude, threshold=threshold)
  cv2.imwrite(f"out/thresh_edges_{image_name}", thresholded_magnitude)

  # magnitude = ip.extract_edges(gray, "CGRAD")
  # enhanced_thresholded_magnitude = ip.segment_by_threshold(enhanced_magnitude, threshold=threshold)
  # cv2.imwrite(f"out/enhanced_thresholded_magnitude_{image_name}", enhanced_thresholded_magnitude)

  confirmed_detections = []
  detection_index = 0

  for x, y, width, height in vj_detections:
    print("--- Analizing VJ detection:", detection_index)

    detection_region = gray[y : y+height, x : x+width]
    detection_region = ip.resize_with_aspect_ratio(detection_region, max_side=150)
    
    # cv2.imwrite(f'preprocess/detected_region_{detection_index}.jpg', detection_region)
    # print("--> Detection: ", detection_index)

    circle_detector = HoughTransformCircles(
      detection_region,
      min_radius=25,
      max_radius=100,
      image_name=image_name,
      edge_detection_strategy="GRAD",
    )
    circle_detector.process_space(threshold=threshold)
    circles = circle_detector.detect_circles(minimum_votes=14)
    num_of_circles = len(circles)
    print("Detected Circles", num_of_circles)
    
    if (num_of_circles >= 2 and num_of_circles <= 4):
      confirmed_detections.append((x, y, width, height))
    else:
      line_detector = HoughTransformLines(detection_region)
      line_detector.process_space(threshold=threshold)
      lines = line_detector.detect_lines(minimum_votes=50)
      num_of_lines = len(lines)
      print("Detected Lines", num_of_lines)
      if ((num_of_lines >= 18 and num_of_lines <= 20) or (num_of_circles == 1 and num_of_lines >= 13)):
        confirmed_detections.append((x, y, width, height))

    detection_index += 1

  for x, y, width, height in confirmed_detections:
    image = cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

  util.print_report(ground_truth_dartboards, confirmed_detections)

  cv2.imwrite(f'out/labeled_{image_name}', image)
  util.show_image(image)
