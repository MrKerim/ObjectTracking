# import the necessary packages
from ultralytics import YOLO
import numpy as np

import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import cv2


model = YOLO("yolov8m.pt")
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]
# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
	tracker = cv2.Tracker_create(args["tracker"].upper())
# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
	# initialize a dictionary that maps strings to their corresponding
	# OpenCV object tracker implementations
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"mil": cv2.TrackerMIL_create
	}
	# grab the appropriate object tracker using our dictionary of
	# OpenCV object tracker objects
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
# initialize the bounding box coordinates of the object we are going
# to track
initBB = None



##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################


# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
# initialize the FPS throughput estimator
fps = None

# loop over frames from the video stream
cntr = 0
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	# check to see if we have reached the end of the stream
	if frame is None:
		break
	# resize the frame (so we can process it faster) and grab the
	# frame dimensions
	frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]
	
# check to see if we are currently tracking an object
	cntr += 1
	if initBB is not None:
		# grab the new bounding box coordinates of the object
		(success, box) = tracker.update(frame)
		# check to see if the tracking was a success
		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h),
				(0, 255, 0), 2)
		else:
			print("tracker failed Initilizing YOLO")
			result = model(frame)
			bboxes = np.array(result[0].boxes.xyxy.cpu(),dtype = int)
			classes = np.array(result[0].boxes.cls.cpu(),dtype = int)
			(x,y,x2,y2) = (0,0,0,0)
			for bbox, cls in zip(bboxes, classes):
				if cls == 4:
					(x,y,x2,y2) = bbox
					tracker = cv2.TrackerKCF_create()
					print("x y ... ->",x,y,x2-x,y2-y)
					tracker.init(frame, (x,y,x2-x,y2-y))
					break
			if((x,y,x2,y2)==(0,0,0,0)):
				print("Yolo failed")
            # start OpenCV object tracker using the supplied bounding box
		    # coordinates, then start the FPS throughput estimator as well
			


		# update the FPS counter
		fps.update()
		fps.stop()
		# initialize the set of information we'll be displaying on
		# the frame
		info = [
			("Tracker", args["tracker"]),
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps.fps())),
		]
		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 's' key is selected, we are going to "select" a bounding
	# box to track
	
	if key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		
		result = model(frame)
		bboxes = np.array(result[0].boxes.xyxy.cpu(),dtype = int)
		classes = np.array(result[0].boxes.cls.cpu(),dtype = int)
		for bbox, cls in zip(bboxes, classes):
			if cls == 4:
				(x,y,x2,y2) = bbox
				initBB = (x,y,x2-x,y2-y)
				print("initBB: ",initBB)
				tracker = cv2.TrackerKCF_create()
				tracker.init(frame, initBB)
				break
        # start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
		fps = FPS().start()
		
# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break
# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()
# otherwise, release the file pointer
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()