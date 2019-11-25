import cv2
import json
import numpy as np

def abs(n):
  return n if n >= 0 else -n

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

def show_image(image, title="Image"):
  cv2.imshow(title, image)
  cv2.moveWindow(title, 0, 20);
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def load_ground_truth(ground_truth_file):
  json_string = open(ground_truth_file).read()
  return json.loads(json_string)

def segment_by_threshold(image, threshold):
  result = np.empty(image.shape)
  for row in range(image.shape[0]):
    for col in range(image.shape[1]):
      if image[row, col] >= threshold:
        result[row, col] = image[row, col]
  return result

def normalize(image, min_value, max_value):
  normalized = np.empty(image.shape)
  return cv2.normalize(image, normalized, min_value, max_value, cv2.NORM_MINMAX)

def image_bounds(image, pixel_value):
  width = np.size(image, 0); height = np.size(image,1)
  pixel_bound_x = pixel_value ; pixel_bound_y = pixel_value; scale_percent = 1

  if (width > pixel_bound_x or height > pixel_bound_y):
    ratioX = (pixel_bound_x/width)
    ratioY = (pixel_bound_y/height)
    if (width > height):
      scale_percent = ratioX
    else: 
      scale_percent = ratioY

  width = int(image.shape[1] * scale_percent)
  print("New width:", width)
  height = int(image.shape[0] * scale_percent)
  print("New height:", height)

  dim = (width, height)
  resized_gray = cv2.resize(image, dim, cv2.INTER_AREA)

  return resized_gray

def viola_jones_detection(cascade_file, gray_image):
  cascade_classifier = cv2.CascadeClassifier(cascade_file)
  return cascade_classifier.detectMultiScale(gray_image, 1.1, 2, 0, (50, 50), (500, 500))

def viola_jones_cascade(cascade_file, image_name,image,resized_gray):

  #Multicascade  Detection
  images_to_dartboards = load_ground_truth("../task2/dartboards.json")
  ground_truth_dartboards = images_to_dartboards[image_name]
  number_of_dartboards = len(ground_truth_dartboards)
  print("Number of dartboards:", number_of_dartboards)
  
  vj_dartboards = cv2.CascadeClassifier('../task2/dartcascade/cascade.xml')
  detected_dartboards = vj_dartboards.detectMultiScale(resized_gray, 1.1, 2, 0, (50, 50), (500, 500))
  number_of_detections = len(detected_dartboards)
  print("Number of detections:", number_of_detections)
  

  for (x1, y1, x2, y2) in ground_truth_dartboards:
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

  for (x, y, w, h) in detected_dartboards:
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

  threshold = 0.5
  succeded_detections = 0
  i = 0

  for (x1, y1, x2, y2) in ground_truth_dartboards:
    ground_truth_p1 = (x1, y1)
    ground_truth_p2 = (x2, y2)
    for (x, y, w, h) in detected_dartboards:
      detected_p1 = (x, y)
      detected_p2 = (x + w, y + h)
      iou = intersection_over_union(ground_truth_p1, ground_truth_p2, detected_p1, detected_p2)
      if(iou >= threshold):
        succeded_detections += 1

  print("True Positives:", succeded_detections)
  false_positives = number_of_detections - succeded_detections
  print("False Positives:", false_positives)
  recall = succeded_detections / number_of_dartboards # Also called sensitivity and recall
  print("True Positive Rate: {} / {} = ".format(succeded_detections, number_of_dartboards), recall) 
  precision = succeded_detections / (succeded_detections + false_positives)  # Also called Positive Predictive Value (PPV)
  print("Precision:", precision)
  f1 = (2 * (recall * precision)) / (recall + precision)
  print("F1:", f1)

  # htc = HoughTransformCircles(mag, dir)
  # htc.process_space()
  # circles = htc.detect_circles()
  


