import cv2
import numpy as np
import sys
import os
import math
from random import randint
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH); 
import util
import image_processing as ip

class HoughTransformCircles(object):
  def __init__(
    self,
    image,
    min_radius=30,
    max_radius=100,
    image_name=None,
    edge_detection_strategy="GRAD",
  ):
    self.image = image
    self.height = image.shape[0]
    self.width = image.shape[1]
    self.min_radius = min_radius
    self.max_radius = max_radius
    self.edges = None
    self.direction = None
    self.h_space = None
    self.image_name = image_name or str(randint(16, 100)) + ".jpg"
    self.edge_detection_strategy = edge_detection_strategy

  def process_space(self, threshold):
    self.__process_gradient_magnitude()
    self.__process_gradient_direction()
    possible_radious = self.max_radius - self.min_radius
    self.h_space = np.zeros([self.height, self.width, possible_radious])
    for row in range(self.height):
      for col in range(self.width):
        if self.edges[row][col] > threshold:
          for r_index in range(possible_radious):
            radius = int(self.min_radius + r_index)
            direction_in_radians = (self.direction[row][col] * 2 * np.pi) / 255
            a = int(radius * math.cos(direction_in_radians))
            b = int(radius * math.sin(direction_in_radians))

            y0 = row - b
            x0 = col - a
            if (self.__inside_image_scope(y = y0, x = x0)):
              self.h_space[y0][x0][r_index] += 1

            y0 = row + b
            x0 = col + a
            if (self.__inside_image_scope(y = y0, x = x0)):
              self.h_space[y0][x0][r_index] += 1
    print("HTC: Maximum number of votes:", np.max(self.h_space))
    return self.h_space
  
  def squash_space(self, scale):
    h_space_2d = np.sum(self.h_space, axis=2) * scale
    print("HTC: Maximum number of votes in 2D:", np.max(h_space_2d))
    # cv2.imwrite(f'h_space/circles_{self.image_name}', h_space_2d)
    return h_space_2d

  def detect_circles(self, minimum_votes=20):
    detections = []; circles = []
    for row in range(self.h_space.shape[0]):
      for col in range(self.h_space.shape[1]):
        for r_index in range(self.h_space.shape[2]):
          if (self.h_space[row, col, r_index] >= minimum_votes):
            radius = r_index + self.min_radius
            detections.append((row, col, radius))
    for row, col, radius in detections:
      if all( (abs(radius - c_rad) > 12 or abs(row - c_row) > 25 or abs(col - c_col) > 25) for c_row, c_col, c_rad in circles):
        circles.append((row, col, radius))
    return circles

  def __inside_image_scope(self, x, y):
    return y >= 0 and x >= 0 and y < self.height and x < self.width

  def __process_gradient_magnitude(self):
    self.edges = ip.extract_edges(self.image, self.edge_detection_strategy)
    # cv2.imwrite(f'out/edges_{self.image_name}', self.edges)

  def __process_gradient_direction(self):
    self.direction = ip.gradient_direction(self.image)
    # cv2.imwrite(f'out/dir_{self.image_name}', self.direction)


if __name__ == "__main__":
  print("Main: Hough Transform Circles")
  image_path = sys.argv[1]
  image_name = image_path.split('/')[-1]
  print(f"Opening {image_name}")

  image = cv2.imread(image_path)
  
  # max_side = 150
  # image = ip.resize_with_aspect_ratio(image, max_side)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  min_radius = 20
  max_radius = 100
  htc = HoughTransformCircles(
    image=gray,
    min_radius=min_radius,
    max_radius=max_radius,
    image_name=image_name,
    edge_detection_strategy="GRAD"
  )
  threshold = 50
  h_space = htc.process_space(threshold=threshold)
  h_space_2d = htc.squash_space(scale=1)

  circles = htc.detect_circles(minimum_votes=14)
  print("Found Circles:", len(circles))
  for row, col, radius in circles:
    p0 = (col - radius, row - radius)
    p1 = (col + radius, row + radius)
    image = cv2.rectangle(image, p0, p1, (0, 255, 0), 1)

  cv2.imwrite(f'out/circles_{image_name}', image)
  util.show_image(image)
