# Description : Testing YOLO with OpenCV
# Importing necessary libraries
import numpy as np
import cv2
from ultralytics import YOLO
import imutils
from imutils.video import VideoStream

model = YOLO("yolov8m.pt")

# Capturing the video from source
cap = cv2.VideoCapture("test2.mp4")
names = model.names
# We use a while loop to iterate through the frames of the video
while True:
    # Reading the frame ret is a boolean value which returns true if the frame is read correctly
    # frame is the frame itself
    ret , frame = cap.read()
    if not ret:
        break



    result = model(frame,device = "mps")
    bboxes = np.array(result[0].boxes.xyxy.cpu(),dtype = int)
    classes = np.array(result[0].boxes.cls.cpu(),dtype = int)
    for bbox, cls in zip(bboxes, classes):
        (x,y,x2,y2) = bbox
        print(cls)
        cv2.rectangle(frame,(x,y),(x2,y2),(0,0,255),2)
        cv2.putText(frame,names[int(cls)],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    
    # With imshow we display the frame
    cv2.imshow("frame",frame)
    # We use waitkey to wait for a key press
    key = cv2.waitKey(1)
    # If the key pressed is q then we break out of the loop
    if key == ord("q"):
        break
# We release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
