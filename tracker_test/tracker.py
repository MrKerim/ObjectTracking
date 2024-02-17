import matplotlib.pyplot as plt
import cv2
import numpy as np
def rgb2YCbCr(tensor):
    transformation_matrix = np.array([[0.299, 0.587, 0.114],[-0.1687, -0.3313, 0.5],[0.5, -0.4187, -0.0813]])
    H, W = tensor.shape[:2]
    reshaped_tensor = tensor.reshape(H*W, 3)
    ycbcr = np.dot(reshaped_tensor, transformation_matrix.T)
    ycbcr = ycbcr.reshape(H, W, 3)
    return ycbcr

def track():  
    initBB = None
    cap = cv2.VideoCapture("Track_car.mov")
    while(True):
        ret, frame = cap.read()
    
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



        cv2.imshow("Frame",frame)
    

if __name__ == "__main__":
    track()