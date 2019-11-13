import cv2
import json 

def load_ground_truth(ground_truth_file):
  json_string = open(ground_truth_file).read()
  return json.loads(json_string)

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

def overlap(p1, p2, list_of_faces):
  for (x, y, w, h) in list_of_faces:
    detected_p1 = (x, y)
    detected_p2 = (x + w, y + h)
    if (p1[0] < detected_p2[0] and p2[0] > detected_p1[0] and
        p1[1] < detected_p2[1] and p2[1] > detected_p1[1]):
      return (detected_p1, detected_p2)
  return (None, None)

# Main

image_name = "dart5.jpg"
image_path = "in/{}".format(image_name)
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

images_to_faces = load_ground_truth("faces.json")
faces_list = images_to_faces[image_name]
number_of_faces = len(faces_list)
print("Number of faces: ", number_of_faces)

face_cascade = cv2.CascadeClassifier('frontalface.xml')
detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
number_of_detections = len(detected_faces)
print("Number of detections: ", number_of_detections)

threshold = 0.7
succeded_detections = 0

for face_coordinate in faces_list:
  ground_truth_p1 = (face_coordinate[0], face_coordinate[1])
  ground_truth_p2 = (face_coordinate[2], face_coordinate[3])
  detected_p1, detected_p2 = overlap(ground_truth_p1, ground_truth_p2, detected_faces)
  # print(detected_p1, detected_p2)
  if(detected_p1 != None and detected_p2 != None):
    iou = intersetion_over_union(ground_truth_p1, ground_truth_p2, detected_p1, detected_p2)
    print("{} > {}".format(iou, threshold))
    if(iou >= threshold):
      succeded_detections += 1

print("Number of successes: ", succeded_detections)
print("Success rate: {} / {} = ".format(succeded_detections, number_of_faces),
  succeded_detections / number_of_faces)

# img = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# cv2.imshow('img', image)
# cv2.imwrite('out/GT_dart15.png',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()