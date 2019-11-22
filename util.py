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
