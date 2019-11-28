import cv2
import json
import numpy as np

def rectangle_area(p1, p2):
  return (p2[0] - p1[0] + 1) * (p2[1] - p1[1] + 1)

def overlap(p1, p2, q1, q2):
  return (p1[0] < q2[0] and p2[0] > q1[0] and
          p1[1] < q2[1] and p2[1] > q1[1])

def intersection_over_union(ground_truth_p1, ground_truth_p2, detected_p1, detected_p2):
  ground_truth_area = rectangle_area(ground_truth_p1, ground_truth_p2)
  detected_area = rectangle_area(detected_p1, detected_p2)
  if (overlap(ground_truth_p1, ground_truth_p2, detected_p1, detected_p2)):
    intersection_p1 = (max(ground_truth_p1[0], detected_p1[0]), max(ground_truth_p1[1], detected_p1[1]))
    intersection_p2 = (min(ground_truth_p2[0], detected_p2[0]), min(ground_truth_p2[1], detected_p2[1]))
    intersection_area = rectangle_area(intersection_p1, intersection_p2)
  else:
    intersection_area = 0
  union_area = ground_truth_area + detected_area - intersection_area
  return intersection_area / union_area

def print_report(ground_truth, detections, iou_threshold=0.5):
  print("== Permormance Report ==")
  number_of_grounds_truths = len(ground_truth)
  print("Number of ground truths:", number_of_grounds_truths)
  number_of_detections = len(detections)
  print("Number of detections:", number_of_detections)

  succeded_detections = 0
  for (x1, y1, x2, y2) in ground_truth:
    ground_truth_p1 = (x1, y1)
    ground_truth_p2 = (x2, y2)
    for (x, y, width, height) in detections:
      detected_p1 = (x, y)
      detected_p2 = (x + width, y + height)
      iou = intersection_over_union(ground_truth_p1, ground_truth_p2, detected_p1, detected_p2)
      if(iou >= iou_threshold):
        succeded_detections += 1
  
  print("True Positives:", succeded_detections)
  false_positives = number_of_detections - succeded_detections
  print("False Positives:", false_positives)
  tpr = succeded_detections / number_of_grounds_truths # Also called sensitivity and recall
  print("True Positive Rate:", tpr)
  
  try:
    precision = succeded_detections / (succeded_detections + false_positives)  # Also called Positive Predictive Value (PPV)
  except Exception:
    precision = 0
  print("Precision:", precision)

  try:  
    f1 = (2 * (tpr * precision)) / (tpr + precision)
  except Exception:
    f1 = 0
  print("F1:", f1)

def show_image(image, title="Image"):
  cv2.imshow(title, image)
  cv2.moveWindow(title, 0, 20);
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def load_ground_truth(ground_truth_file):
  json_string = open(ground_truth_file).read()
  return json.loads(json_string)

def viola_jones_detection(cascade_file, gray_image):
  cascade_classifier = cv2.CascadeClassifier(cascade_file)
  return cascade_classifier.detectMultiScale(gray_image, 1.1, 1, 0, (50, 50), (500, 500))
