import cv2
import json 

def load_ground_truth(ground_truth_file):
  json_string = open(ground_truth_file).read()
  return json.loads(json_string)

def rectangle_area(p1, p2):
  return (p2[0] - p1[0] + 1) * (p2[1] - p1[1] + 1)

def overlap(p1, p2, q1, q2):
  return (p1[0] < q2[0] and p2[0] > q1[0] and
          p1[1] < q2[1] and p2[1] > q1[1])

def intersection_over_union(ground_truth_p1, ground_truth_p2, detected_p1, detected_p2):
  ground_truth_area = rectangle_area(ground_truth_p1, ground_truth_p2)
  detected_area = rectangle_area(detected_p1, detected_p2)
  if (overlap(ground_truth_p1, ground_truth_p2, detected_p1, detected_p2)):
    intersection_p1 = (max(ground_truth_p1[0], detected_p1[0]), max(ground_truth_p1[1], detected_p1[1]))
    intersection_p2 = (min(ground_truth_p2[0], detected_p2[0]), min(ground_truth_p2[1], detected_p2[1]))
    intersection_area = rectangle_area(intersection_p1, intersection_p2)
  else:
    intersection_area = 0
  union_area = ground_truth_area + detected_area - intersection_area
  return intersection_area / union_area

# Main

image_name = "dart5.jpg"
image_path = "in/{}".format(image_name)
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

images_to_faces = load_ground_truth("faces.json")
ground_truth_faces = images_to_faces[image_name]
number_of_faces = len(ground_truth_faces)
print("Number of faces: ", number_of_faces)

face_cascade = cv2.CascadeClassifier('frontalface.xml')
detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
number_of_detections = len(detected_faces)
print("Number of detections: ", number_of_detections)

for faces_coordinates in ground_truth_faces:
  x1 = faces_coordinates[0]
  y1 = faces_coordinates[1]
  x2 = faces_coordinates[2]
  y2 = faces_coordinates[3]
  image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in detected_faces:
  image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

threshold = 0.5
succeded_detections = 0
i = 0

for face_coordinate in ground_truth_faces:
  ground_truth_p1 = (face_coordinate[0], face_coordinate[1])
  ground_truth_p2 = (face_coordinate[2], face_coordinate[3])
  # print("Ground truth", ground_truth_p1, ground_truth_p2)
  for (x, y, w, h) in detected_faces:
    detected_p1 = (x, y)
    detected_p2 = (x + w, y + h)
    iou = intersection_over_union(ground_truth_p1, ground_truth_p2, detected_p1, detected_p2)
    # print(detected_p1, detected_p2, iou)
    if(iou >= threshold):
      succeded_detections += 1

print("Number of successes: ", succeded_detections)
print("True Positive Rate: {} / {} = ".format(succeded_detections, number_of_faces),
  succeded_detections / number_of_faces)

# img = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# cv2.imshow('img', image)
labeled_file_name = image_name.replace("dart", "out/labeled")
cv2.imwrite(labeled_file_name, image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()