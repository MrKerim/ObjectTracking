import cv2

tracker = cv2.TrackerKCF_create()
cap = cv2.VideoCapture("test2.mp4")

initBB = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("s"):
        initBB = cv2.selectROI("frame",frame,fromCenter=False,showCrosshair=True)
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame,initBB)
    if initBB is not None:    
        (success, box) = tracker.update(frame)
        if success:
            print(box)
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
    cv2.imshow("frame",frame)
