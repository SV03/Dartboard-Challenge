import cv2
import sys
import os
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH); import util

if __name__ == '__main__':
  image_path = sys.argv[1]
  image_name = image_path.split('/')[-1]
  print(f"Opening {image_name}")
  image = cv2.imread(image_path)
  # magnitude = calculate the magniture
  # segmented_magnitude = segment by threshold

  # util.show_image(segmented_magnitude)

