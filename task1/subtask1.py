import cv2
import sys
import os
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_PATH);
import util

if __name__ == '__main__':
  image_path = sys.argv[1]
  image_name = image_path.split('/')[-1]
  print(f"Opening {image_name}")
  image = cv2.imread(image_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  images_to_faces = util.load_ground_truth("faces.json")
  ground_truth_faces = images_to_faces[image_name]
  number_of_faces = len(ground_truth_faces)
  print("Number of faces:", number_of_faces)

  face_cascade = cv2.CascadeClassifier('frontalface.xml')
  detected_faces = face_cascade.detectMultiScale(gray, 1.1, 1, 0, (50, 50), (500, 500))
  number_of_detections = len(detected_faces)
  print("Number of detections:", number_of_detections)

  for faces_coordinates in ground_truth_faces:
    x1 = faces_coordinates[0]
    y1 = faces_coordinates[1]
    x2 = faces_coordinates[2]
    y2 = faces_coordinates[3]
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

  for (x, y, w, h) in detected_faces:
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

  iou_threshold = 0.5

  util.print_report(ground_truth_faces, detected_faces, iou_threshold)

  # labeled_file_name = image_name.replace("dart", "out/labeled")
  # cv2.imwrite(labeled_file_name, image)

  util.show_image(image)
