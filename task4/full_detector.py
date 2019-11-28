import cv2
import numpy as np
import os
import sys
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH);
import util
import image_processing as ip

if __name__ == "__main__":
  print("Main: Full detector")
  image_path = sys.argv[1]
  image_name = image_path.split('/')[-1]
  print(f"Opening {image_name}")
  image = cv2.imread(image_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  edges = cv2.Canny(gray, 100, 200)
  contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
  print(len(contours))
  cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

  util.show_image(image)
