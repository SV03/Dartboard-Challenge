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
    self.threshold = threshold
    self.min_radius = min_radius
    self.max_radius = max_radius
    
    possibleRadious = max_radius - min_radius
    h_space_padding = 143
    h_space = np.zeros([2000, 2000, possibleRadious])
    min_x = 99999
    max_x = -99999
    min_y = 99999
    max_y =-99999
    for row in range (self.direction.shape[0]):
      for col in range (self.direction.shape[1]):
        #print("Passed Loop", col) 
        #print(self.magnitude[row][col])
        if self.magnitude[row][col] > threshold:
          #print("yes")
          for r in range (possibleRadious):
            #print(possibleRadious)
            radius = min_radius + r
            direction_in_radians = self.direction[row][col] * 0.0243
            x0 = col - radius * math.cos(direction_in_radians)
            y0 = row - radius * math.sin(direction_in_radians)
            if (x0 > max_x): max_x = x0
            if (x0 < min_x): min_x = x0
            if (y0 > max_y): max_y = y0
            if (y0 < min_y): min_y = y0
            h_space[int(x0 + h_space_padding)][int(y0 + h_space_padding)][int(r)]
    print("Max X", max_x)
    print("Min X", min_x)
    print("Max Y", max_y)
    print("Min Y", min_y)
    h_space_width = -min_x + max_x
    print("Width:",h_space_width)
    h_space_height  = -min_y + max_y
    print("Height:",h_space_height)
    return h_space.shape

if __name__ == "__main__":
  image_path = sys.argv[1]
  image_name = image_path.split('/')[-1]
  print(f"Opening {image_name}")
  image = cv2.imread(image_path)
  file_number = 1
  grad_magnitude = cv2.imread(f'out/segmented{file_number}.jpg')
  print("Pass1")
  gray_magnitude = cv2.cvtColor(grad_magnitude, cv2.COLOR_BGR2GRAY)
  print("Pass2")
  grad_direction = cv2.imread(f'out/gradient_dir{file_number}.jpg')
  
  print("Pass3")
  gray_direction = cv2.cvtColor(grad_direction, cv2.COLOR_BGR2GRAY)
  print("Pass4")
  
  htl = HoughTransformCircles(gray_magnitude, gray_direction)
  print("Pass5")

  h_space = htl.process(threshold=250, min_radius=25, max_radius=145)
  print(h_space)


  # labeled_file_name = image_name.replace(image_name, "out/labeled_h_space")
  # cv2.imwrite(labeled_file_name, image)


