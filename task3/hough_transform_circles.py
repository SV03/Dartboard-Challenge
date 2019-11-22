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

  def process(self, threshold = 250, min_radius = 20, max_radius = 145):
    possible_radious = max_radius - min_radius
    h_space_padding = 140
    h_space = np.zeros([1500, 1500, possible_radious])
    min_x = 99999; max_x = -99999
    min_y = 99999; max_y =-99999
    for row in range (self.direction.shape[0]):
      for col in range (self.direction.shape[1]):
        if self.magnitude[row][col] > threshold:
          for r in range (possible_radious):
            radius = min_radius + r
            direction_in_radians = (self.direction[row][col] * 2 * np.pi) / 255
            a = radius * math.cos(direction_in_radians)
            b = radius * math.sin(direction_in_radians)
          
            x0 = col - a
            y0 = row - b
            h_space[int(x0 + h_space_padding)][int(y0 + h_space_padding)][int(r)] += 1
            if (x0 < min_x): min_x = x0
            if (y0 < min_y): min_y = y0

            x0 = col + a
            y0 = row + b
            h_space[int(x0 + h_space_padding)][int(y0 + h_space_padding)][int(r)] += 1
            if (x0 > max_x): max_x = x0
            if (y0 > max_y): max_y = y0
    print("Max X", max_x)
    print("Min X", min_x)
    print("Max Y", max_y)
    print("Min Y", min_y)
    h_space_width = int(-min_x + max_x)
    print("Width:",h_space_width)
    h_space_height  = int(-min_y + max_y)
    print("Height:",h_space_height)
    return np.sum(h_space, axis=2) * 3

if __name__ == "__main__":
  file_number = 14

  grad_magnitude = cv2.imread(f'out/edges{file_number}.jpg')
  gray_magnitude = cv2.cvtColor(grad_magnitude, cv2.COLOR_BGR2GRAY)

  grad_direction = cv2.imread(f'out/gradient_dir{file_number}.jpg')
  gray_direction = cv2.cvtColor(grad_direction, cv2.COLOR_BGR2GRAY)

  htl = HoughTransformCircles(gray_magnitude, gray_direction)

  h_space = htl.process(threshold=250, min_radius=25, max_radius=145)
  print(h_space.shape)

  cv2.imwrite(f'h_space/circles{file_number}.jpg', h_space)


