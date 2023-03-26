from Libraries.hand_module import *
import cv2
import numpy as np
import math



class handLib:
    def __init__(self):
        pass

    def coords(self, y1, y2, x1, x2):
        return Coords(y1,y2,x1,x2)

    def hand_entry(self, last_frame, current_frame, hand_roi, border):
        return HandEntry(last_frame, current_frame, hand_roi, border)


    def calc_velocity(self, hand_entry:HandEntry, gpu=False):
        if not gpu:
            last, current = hand_entry.get_grayscale_roi()
            hsv = np.zeros((last.shape[0], last.shape[1], 1))
            # possible resize for performance issues
            flow = cv2.calcOpticalFlowFarneback(
                last, current, None, 0.5, 5, 14, 3, 7, 1.5, 0,
            )
            magnitude, angle = cv2.cartToPolar(
                flow[..., 0], flow[..., 1], angleInDegrees=True,
            )
            # set value according to the normalized magnitude of optical flow
            hsv[..., 0] = cv2.normalize(
                magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX, -1,
            )
            hsv_8u = np.uint8(hsv * 255.0)
            hsv_8u = cv2.resize(hsv_8u,(10,10))
            avg = 0
            for x in np.nditer(hsv_8u):
                if x > 30:
                    avg+=x
            avg /= 100
            return hsv_8u, avg
