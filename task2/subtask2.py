from PIL import Image
import cv2
import sys
import json

def show_image(image, title="Image"):
  cv2.imshow(title, image)
  cv2.moveWindow(title, 0, 20);
  cv2.waitKey()
  cv2.destroyAllWindows()

def load_ground_truth(ground_truth_file):
  json_string = open(ground_truth_file).read()
  return json.loads(json_string)

image_path = sys.argv[1]
image_name = image_path.split('/')[-1]
print(f"Opening {image_name}")
image = cv2.imread(image_path)

images_to_dartboards = load_ground_truth("dartboards.json")
ground_truth_dartboards = images_to_dartboards[image_name]
number_of_dartboards = len(ground_truth_dartboards)
print("Number of dartboards:", number_of_dartboards)

for (x1, y1, x2, y2) in ground_truth_dartboards:
  image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

show_image(image)
