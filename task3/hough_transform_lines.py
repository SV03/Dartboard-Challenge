import numpy as np
import cv2
import os
import sys
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH); import util

class HoughTransformLines(object):
  def __init__(self, grad_magnitude):
    self.grad_magnitude = grad_magnitude

  def process(self, threshold=200):
    h_space = np.zeros((1800, 360))
    max_roe = -9999999999; min_roe = 9999999999
    roe_padding = 800
    for row in range(self.grad_magnitude.shape[0]):
      for col in range(self.grad_magnitude.shape[1]):
        if self.grad_magnitude[row][col] >= threshold:
          for theta in range(360):
            theta_radians = (theta * 2 * np.pi) / 360
            roe = int(col * np.cos(theta_radians) + row * np.sin(theta_radians))
            if roe > max_roe: max_roe = roe
            if roe < min_roe: min_roe = roe
            h_space[roe + roe_padding, theta] += 1
    # print("max_roe", max_roe); print("min_roe", min_roe)
    # normalized = np.zeros(h_space.shape)
    # cv2.normalize(h_space, normalized, 0, 255, cv2.NORM_MINMAX)
    h_space = h_space * 4 # Make curves brighter
    h_space = cv2.resize(h_space, (550, 650))
    return cv2.hconcat([h_space, h_space])

if __name__ == "__main__":
  file_number = 0
  grad_magnitude = cv2.imread(f'out/segmented{file_number}.jpg')
  gray_grad_magnitude = cv2.cvtColor(grad_magnitude, cv2.COLOR_BGR2GRAY)
  
  htl = HoughTransformLines(gray_grad_magnitude)
  h_space = htl.process(threshold=255)

  cv2.imwrite(f'h_space/lines{file_number}.jpg', h_space)
  # util.show_image(h_space)
