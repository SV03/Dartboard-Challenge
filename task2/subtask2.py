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
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  images_to_dartboards = util.load_ground_truth("dartboards.json")
  ground_truth_dartboards = images_to_dartboards[image_name]
  number_of_dartboards = len(ground_truth_dartboards)

  cascade_file = 'dartcascade/cascade.xml'
  detected_dartboards = util.viola_jones_detection(cascade_file, gray)
  number_of_detections = len(detected_dartboards)

  for (x1, y1, x2, y2) in ground_truth_dartboards:
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

  for (x, y, w, h) in detected_dartboards:
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

  iou_threshold = 0.5
  util.print_report(ground_truth_dartboards, detected_dartboards, iou_threshold)

  labeled_file_name = image_name.replace("dart", "out/labeled_dart")
  cv2.imwrite(labeled_file_name, image)

  util.show_image(image)
