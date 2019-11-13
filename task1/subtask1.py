import cv2
import json 

def rectangle_area(p1, p2):
  return (p2[0] - p1[0] + 1) * (p2[1] - p1[1] + 1)

def intersetion_over_union(ground_truth_p1, ground_truth_p2, detected_p1, detected_p2):
  intersection_p1 = (max(ground_truth_p1[0], detected_p1[0]), max(ground_truth_p1[1], detected_p1[1]))
  intersection_p2 = (min(ground_truth_p2[0], detected_p2[0]), max(ground_truth_p2[1], detected_p2[1]))
  intersection_area = rectangle_area(intersection_p1, intersection_p2)
  ground_truth_area = rectangle_area(ground_truth_p1, ground_truth_p2)
  detected_area = rectangle_area(detected_p1, detected_p2)
  union_area = ground_truth_area + detected_area - intersection_area
  return intersection_area / union_area

# Main

image_name = "dart15.jpg"
image_path = "in/{}".format(image_name)
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Draw ground truth

json_string = open('faces.json').read()
images_to_faces = json.loads(json_string)

faces_list = images_to_faces[image_name]
for faces_coordinates in faces_list:
  x1 = faces_coordinates[0]
  y1 = faces_coordinates[1]
  x2 = faces_coordinates[2]
  y2 = faces_coordinates[3]
  img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Cascade Detection

face_cascade = cv2.CascadeClassifier('frontalface.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
  img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv2.imshow('img', img)
# cv2.imwrite('out/GT_dart15.png',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

ground_truth_p1 = (1, 1)
ground_truth_p2 = (3, 3)
detected_p1 = (2, 2)
detected_p2 = (3, 4)

print(intersetion_over_union(ground_truth_p1, ground_truth_p2, detected_p1, detected_p2))
