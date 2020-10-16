# Social-distance-detection-system
This social distancing detection tool detects whether people are keeping safe distance from each other.
Programming language used : PYTHON
# Requirements:
We need to import certain python libraries like
* numpy
* cv2
# Therefore we need to install them using commands like
 * pip install numpy
 * pip install opencv
# Working:
* Detect the humans in the frame with yolov4-tiny convolutional neural network.
* Calculate the distance between all the instances of humans detected in the frame.
* If this application detects more than 4 people violating the rule, it sends a ALERT message to the security guard who is in-charge of that place.
