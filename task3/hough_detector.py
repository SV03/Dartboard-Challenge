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

# def gradient_direction(image):
#   result = np.empty(image.shape)
#   gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#   gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#   # result = np.arctan(gradient_y / gradient_x)
#   for row in range(image.shape[0]):
#     for col in range(image.shape[1]):
#       # if gradient_x[row, col] !=0:
#       #   division = gradient_y[row, col] / gradient_x[row, col]
#       # else:
#       #   print("Zero division")
#       #   division = 99999999999999999
#       # result[row, col] = np.arctan(division)
#       result[row, col] = math.atan2(gradient_x[row, col], gradient_y[row, col])
#   return result

if __name__ == '__main__':
  image_path = sys.argv[1]
  image_name = image_path.split('/')[-1]
  print(f"Opening {image_name}")
  image = cv2.imread(image_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cv2.imwrite('out/gray_{}'.format(image_name), gray)

  magnitude = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
  threshold = 215
  segmented_magnitude = segment_by_threshold(magnitude, threshold)
  # gradient_direction = gradient_direction(gray)

  labeled_file_name = image_name.replace("dart", "out/magnitude")
  cv2.imwrite(labeled_file_name, magnitude)
  labeled_file_name = image_name.replace("dart", "out/segmented")
  cv2.imwrite(labeled_file_name, segmented_magnitude)
  # labeled_file_name = image_name.replace("dart", "out/g_direction")
  # cv2.imwrite(labeled_file_name, gradient_direction)
  util.show_image(segmented_magnitude)
