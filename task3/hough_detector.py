import cv2
import numpy as np
import sys
import os
import math
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH); 
import util
import image_processing as ip

def gradient_direction(image):
  direction = np.empty(image.shape)
  normalized = np.empty(direction.shape)
  gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
  gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
  for row in range(image.shape[0]):
    for col in range(image.shape[1]):
      direction[row, col] = math.atan2(gradient_y[row, col], gradient_x[row, col])
  return cv2.normalize(direction, normalized, 0, 255, cv2.NORM_MINMAX)

if __name__ == '__main__':
  image_path = sys.argv[1]
  image_name = image_path.split('/')[-1]
  print(f"Opening {image_name}")
  image = cv2.imread(image_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  max_side = 300
  image = ip.resize_with_aspect_ratio(image, max_side=max_side)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  cascade_file = '../task2/dartcascade/cascade.xml'
  vj_detections = util.viola_jones_detection(cascade_file, gray)

  for x, y, width, height in vj_detections:
    image = cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    # extract 
    # magnitude = ip.gradient_magnitude()
    # direction = 
    # htc = HougTransformCirle(magnitude, direction)
    # htc.process_space()
    # circles htc.detect_circles( )
    # if (len((circles) >= 1):
      # htl = HougTransformLines()   # 4 -HT Lines
      # if (htl.number_of_lines()):
        # Confirm it is a dartboard
        # else:
        # REJECT DETECTION


  # cv2.imwrite(f'preprocess/grad_{image_name}', magnitude)

  # threshold = 80
  # gradient = ip.segment_by_threshold(magnitude, threshold=threshold)
  # cv2.imwrite(f'preprocess/grad_thre{image_name}', gradient)

  # grad_direction = ip.gradient_direction(gray)
  # cv2.imwrite(f'preprocess/dir_{image_name}', grad_direction)
  # 1 resize for efficiency
  
  # Show image with detections
  # Show Performance Report

  # resized_gray = util.image_bounds(gray, 600)

  # edges = util.segment_by_threshold(laplacian, threshold = 250)
  # labeled_file_name = image_name.replace("dart", "out/edges")
  # cv2.imwrite(labeled_file_name, edges)
  
  # gradient_direction = gradient_direction(resized_gray)
  # labeled_file_name = image_name.replace("dart", "out/gradient_dir")
  # cv2.imwrite(labeled_file_name, gradient_direction)

  util.show_image(image)


