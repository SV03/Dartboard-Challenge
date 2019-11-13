import cv2
import json 

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

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
