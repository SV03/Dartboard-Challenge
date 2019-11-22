import cv2
import numpy as np
import sys
import os
import math
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH); import util

class HoughTransformCircles(object):
  def __init__(self, magnitude, direction):
    self.magnitude = magnitude
    self.direction = direction

  def process_space(self, threshold, min_radius, max_radius, print_report=False):
    possible_radious = max_radius - min_radius
    h_space_padding = 140
    h_space = np.zeros([1000, 1000, possible_radious])
    min_x = 99999; max_x = -99999
    min_y = 99999; max_y =-99999
    for row in range (self.direction.shape[0]):
      for col in range (self.direction.shape[1]):
        if self.magnitude[row][col] > threshold:
          for r in range (possible_radious):
            radius = int(min_radius + r)
            direction_in_radians = (self.direction[row][col] * 2 * np.pi) / 255
            a = int(radius * math.cos(direction_in_radians))
            b = int(radius * math.sin(direction_in_radians))
            
            # x0 = h_space_padding + col
            # y0 = h_space_padding + row
            x0 = + col
            y0 = + row
            if (y0 - b > 0 and  x0 - a > 0):
              h_space[y0 - b][x0 - a][r] += 1
            if (x0 < min_x): min_x = int(x0)
            if (y0 < min_y): min_y = int(y0)

            h_space[y0 + b][x0 + a][r] += 1
            if (x0 > max_x): max_x = int(x0)
            if (y0 > max_y): max_y = int(y0)
    self.h_space = h_space

    if print_report:
      print("Min Y:", min_y)
      print("Max Y:", max_y)
      print("Max X:", max_x)
      print("Min X:", min_x)
      h_space_height = -min_y + max_y
      h_space_width = -min_x + max_x
      print(f"Hough Space Scope - Heigh: {h_space_height}, Width: {h_space_width}")
    return self.h_space
  
  def squash_space(self, scale=1):
    return np.sum(self.h_space, axis=2) * scale

if __name__ == "__main__":
  file_number = 5

  grad_magnitude = cv2.imread(f'out/edges{file_number}.jpg')
  gray_magnitude = cv2.cvtColor(grad_magnitude, cv2.COLOR_BGR2GRAY)

  grad_direction = cv2.imread(f'out/gradient_dir{file_number}.jpg')
  gray_direction = cv2.cvtColor(grad_direction, cv2.COLOR_BGR2GRAY)

  htc = HoughTransformCircles(gray_magnitude, gray_direction)

  h_space = htc.process_space(threshold=253, min_radius=20, max_radius=150, print_report=True)
  print(h_space.shape)
  h_space_2d = htc.squash_space(scale=3)
  print(h_space_2d.shape)

  # h_space = util.normalize(h_space, 0, 255)
  # h_space = util.segment_by_threshold(h_space, 50)
  # print(h_space.shape)

  cv2.imwrite(f'h_space/circles{file_number}.jpg', h_space_2d)
  # util.show_image(grad_magnitude)


