import numpy as np
import cv2
import os
import sys
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH);
import util
import image_processing as ip

class HoughTransformLines(object):
  def __init__(self, image):
    self.grad_magnitude = ip.gradient_magnitude(image)
    self.h_space = None
    cv2.imwrite('out/edges_init.jpg', self.grad_magnitude)

  def process(self, threshold):
    self.h_space = np.zeros((1800, 360))
    # max_roe = -9999999999; min_roe = 9999999999
    roe_padding = 900
    pixels_voting = 0
    for row in range(self.grad_magnitude.shape[0]):
      for col in range(self.grad_magnitude.shape[1]):
        if self.grad_magnitude[row, col] >= threshold:
          pixels_voting += 1
          for theta in range(360):
            theta_radians = (theta * 2 * np.pi) / 360
            roe = int(col * np.cos(theta_radians) + row * np.sin(theta_radians))
            self.h_space[roe + roe_padding, theta] += 1
    print("pixels_voting", pixels_voting)
    self.h_space = cv2.resize(self.h_space, (360, 900))
    return self.h_space

  def detect_lines(self, minimum_votes):
    detections = []; lines = []
    for roe in range(self.h_space.shape[0]):
      for theta in range(self.h_space.shape[1]):
        if (self.h_space[roe, theta] >= minimum_votes):
          detections.append((roe, theta))
    # print("Numb of detections:", len(detections))
    for roe, theta in detections:
      if all( abs(theta - c_theta) > 15 for c_roe, c_theta in lines):
        lines.append((roe, theta))
    return lines

if __name__ == "__main__":
  print("Main: Hough Transform Lines!")
  image_path = sys.argv[1]
  image_name = image_path.split('/')[-1]
  print(f"Opening {image_name}")
  image = cv2.imread(image_path)

  # max_side = 150
  # image = ip.resize_with_aspect_ratio(image, max_side)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  threshold = 150

  htl = HoughTransformLines(gray)
  h_space = htl.process(threshold=threshold)
  lines = htl.detect_lines(minimum_votes=55)
  print("Detected lines:", len(lines))
  print(lines)

  cv2.imwrite(f'h_space/lines_{image_name}', h_space)
  # util.show_image(h_space)
