import cv2
import numpy as np
import sys
import os
import math
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH); import util

def segment_by_threshold(image, threshold):
  result = np.empty(image.shape)
  for row in range(image.shape[0]):
    for col in range(image.shape[1]):
      if image[row, col] >= threshold:
        result[row, col] = image[row, col]
  return result

def gradient_direction(image):
  result = np.empty(image.shape)
  normalized = np.zeros(result.shape)
  gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
  gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
  for row in range(image.shape[0]):
    for col in range(image.shape[1]):
      result[row, col] = math.atan2(gradient_y[row, col], gradient_x[row, col])
  return cv2.normalize(result, normalized, 0, 255, cv2.NORM_MINMAX)

if __name__ == '__main__':
  
  image_path = sys.argv[1]
  image_name = image_path.split('/')[-1]
  print(f"Opening {image_name}")
  image = cv2.imread(image_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  gradient_magnitude = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
  threshold = 215
  segmented_magnitude = segment_by_threshold(gradient_magnitude, threshold)
  gradient_direction = gradient_direction(gray)

  labeled_file_name = image_name.replace("dart", "out/segmented")
  cv2.imwrite(labeled_file_name, segmented_magnitude)
  labeled_file_name = image_name.replace("dart", "out/gradient_dir")
  cv2.imwrite(labeled_file_name, gradient_direction)
  # util.show_image(gradient_direction)
