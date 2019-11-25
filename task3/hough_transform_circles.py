import cv2
import numpy as np
import sys
import os
import math
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH); 
import util
import image_processing as ip

class HoughTransformCircles(object):
  def __init__(self, magnitude, direction, min_radius, max_radius):
    self.magnitude = magnitude
    self.direction = direction
    self.min_radius = min_radius
    self.max_radius = max_radius
    self.height = magnitude.shape[0]
    self.width = magnitude.shape[1]

  def process_space(self, threshold):
    possible_radious = self.max_radius - self.min_radius
    h_space = np.zeros([self.height, self.width, possible_radious])
    for row in range(self.height):
      for col in range (self.width):
        if self.magnitude[row][col] > threshold:
          for r in range (possible_radious):
            radius = int(self.min_radius + r)
            direction_in_radians = (self.direction[row][col] * 2 * np.pi) / 255
            a = int(radius * math.cos(direction_in_radians))
            b = int(radius * math.sin(direction_in_radians))

            y0 = row - b
            x0 = col - a
            if (y0 >= 0 and x0 >= 0 and y0 < self.height and x0 < self.width):
              h_space[y0][x0][r] += 1

            y0 = row + b
            x0 = col + a
            if (y0 >= 0 and x0 >= 0 and y0 < self.height and x0 < self.width):
              h_space[y0][x0][r] += 1
    self.h_space = h_space
    return self.h_space
  
  def squash_space(self, scale):
    return np.sum(self.h_space, axis=2) * scale

  def detect_circles(self, threshold=25):
    detections = []; circles = []
    for row in range(self.h_space.shape[0]):
      for col in range(self.h_space.shape[1]):
        for r_index in range(self.h_space.shape[2]):
          if (self.h_space[row, col, r_index] >= threshold):
            radius = r_index + self.min_radius
            detections.append((row, col, radius))
    for row, col, radius in detections:
      if all( abs(radius - c_rad) > 12 for c_row, c_col, c_rad in circles):
        circles.append((row, col, radius))
    return circles

if __name__ == "__main__":
  image_path = sys.argv[1]
  image_name = image_path.split('/')[-1]
  print(f"Opening {image_name}")

  image = cv2.imread(image_path)
  image = ip.resize_with_aspect_ratio(image, max_side=150)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  edges = ip.gradient_magnitude(gray)
  # cv2.imwrite(f'preprocess/grad_{image_name}', edges)

  threshold = 80
  gradient = ip.segment_by_threshold(edges, threshold=threshold)
  cv2.imwrite(f'preprocess/grad_thre{image_name}', gradient)

  grad_direction = ip.gradient_direction(gray)
  cv2.imwrite(f'preprocess/dir_{image_name}', grad_direction)

  htc = HoughTransformCircles(gradient, grad_direction, min_radius=30, max_radius=150)

  h_space = htc.process_space(threshold=threshold)
  h_space_2d = htc.squash_space(scale=1)
  
  circles = htc.detect_circles(threshold=20)
  for row, col, radius in circles:
    p0 = (col - radius, row - radius)
    p1 = (col + radius, row + radius)
    image = cv2.rectangle(image, p0, p1, (0, 255, 0), 1)

  # cv2.imwrite(f'h_space/circles_{image_name}', h_space_2d)
  cv2.imwrite(f'out/circles_{image_name}', image)
  util.show_image(image)
