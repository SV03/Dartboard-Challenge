# Dartboard-Challenge

### Task 1

```
cd task1
python subtask1.py ../input/dart4.jpg
```
Only the following images can be analyzed: `dart4.jpg`, `dart5.jpg`, `dart13.jpg`, `dart14.jpg`,`dart 15.jpg`.

### Task 2 

For the multiscale we used 1.1 scale_image and 2 min_neighbours.

```
cd task2
python subtask2.py ../input/dart0.jpg
```

#### Train Cascade Classifier
Generate positive dataset with `opencv_createsamples`
```
-maxidev 80 # Maximal intensity deviation of pixels in foreground samples.
-maxxangle 0.8 # max_x_rotation_angle
-maxyangle 0.8 # max_y_rotation_angle
-maxzangle 0.2 # max_z_rotation_angle
```

Train Classifier with `opencv_traincascade`
```
# common arguments
-bg negatives.dat # Background description file.
-numStages 3 #  Number of cascade stages to be trained.
# cascade parameters
-w 20
-h 20
# Boosted classifier parameters
-minHitRate 0.999 #  Minimal desired hit rate for each stage of the classifier. Overall hit rate may be estimated as (min_hit_rate ^ 
-maxFalseAlarmRate 005 # Maximal desired false alarm rate for each stage of the classifier. Overall false alarm rate may be estimated as (max_false_alarm_rate ^ number_of_stagesnumber_of_stages) 
-maxDepth 1 # Maximal depth of a weak tree. A decent choice is 1, that is case of stumps.
-mode ALL # Selects the type of Haar features set used in training. ALL uses the full set of upright and 45 degree rotated feature set
```

`python subtask2.py ../input/dart1.jpg`

### Task 3
A dartboard detector was built by integrating the Viola Jones detector and detecting shapes with Hough Transforms for circles.

#### Hough Transform for circles
`> python task3/hough_transform_circles.py input/circle1.jpg`
