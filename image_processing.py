import cv2
import numpy as np
import math

def normalize(image, min_value, max_value):
  normalized = np.zeros(image.shape)
  return cv2.normalize(image, normalized, min_value, max_value, cv2.NORM_MINMAX, cv2.CV_8UC1)

def segment_by_threshold(image, threshold):
  result = np.empty(image.shape)
  for row in range(image.shape[0]):
    for col in range(image.shape[1]):
      if image[row, col] >= threshold:
        result[row, col] = image[row, col]
  return result

def image_bounds(image, pixel_value):
  width = np.size(image, 0); height = np.size(image, 1)
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

def gradient_direction(image):
  direction = np.empty(image.shape)
  gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
  gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
  for row in range(image.shape[0]):
    for col in range(image.shape[1]):
      direction[row, col] = math.atan2(gradient_y[row, col], gradient_x[row, col])
  # return direction
  return normalize(direction, 0, 255)

def edges(image):
  return cv2.Laplacian(image, cv2.CV_64F, ksize=3)
