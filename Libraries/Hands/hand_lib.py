import math

from Libraries.Hands.hand_module import *
from Libraries.helper import Coords
import cv2
import numpy as np


class handLib:
    def __init__(self):
        self.movement_frames = []
        self.__fps = 1

        self.__gpu_hsv = None
        self.__gpu_hsv_8u = None
        self.__gpu_s = None
        self.__gpu_h = None
        self.__gpu_v = None
        self.__gpu_previous = None
        self.__gpu_current = None

    def __calc_time(self, frame_number, fps):
        return frame_number//fps

    def upload(self, property, value):
        if property == 'fps':
            self.__fps = value
        else:
            print("the specified property doesn't exist")

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
                if x > 100:
                    avg+=x
            avg /= 100
            return hsv_8u, avg
        else:
            last = hand_entry.last_roi
            current = hand_entry.current_roi

            if self.__gpu_previous is None:
                self.__gpu_current = cv2.cuda_GpuMat()
                self.__gpu_previous = cv2.cuda_GpuMat()
            self.__gpu_current.upload(current)
            self.__gpu_previous.upload(last)

            if self.__gpu_hsv is None and self.__gpu_hsv_8u is None:
                self.__gpu_hsv = cv2.cuda_GpuMat(self.__gpu_current.size(), cv2.CV_32FC3)
                self.__gpu_hsv_8u = cv2.cuda_GpuMat(self.__gpu_current.size(), cv2.CV_8UC3)
                self.__gpu_s = cv2.cuda_GpuMat(self.__gpu_current.size(), cv2.CV_32FC1)
                self.__gpu_s.upload(np.ones_like(self.__gpu_current, np.float32))

                self.__gpu_h = cv2.cuda_GpuMat(self.__gpu_current.size(), cv2.CV_32FC1)
                self.__gpu_v = cv2.cuda_GpuMat(self.__gpu_current.size(), cv2.CV_32FC1)

            gpu_last = cv2.cuda.cvtColor(self.__gpu_previous, cv2.COLOR_BGR2GRAY)
            gpu_current = cv2.cuda.cvtColor(self.__gpu_current, cv2.COLOR_BGR2GRAY)
            gpu_flow = cv2.cuda.FarnebackOpticalFlow.create(
                5, 0.5, False, 14, 3, 7, 1.5, 0,
            )
            gpu_flow = cv2.cuda.FarnebackOpticalFlow.calc(
                gpu_flow, gpu_last, gpu_current, None,
            )
            gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
            gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
            cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])
            #  convert from cartesian to polar coordinates to get magnitude and angle
            gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
                gpu_flow_x, gpu_flow_y, angleInDegrees=True,
            )
            self.__gpu_v = cv2.cuda.normalize(gpu_magnitude, 0.0, 1.0, cv2.NORM_MINMAX, -1)
            self.__gpu_v = cv2.cuda.resize(self.__gpu_v, (10, 10))
            value_matrix = self.__gpu_v.download()
            value_matrix = cv2.convertScaleAbs(value_matrix, value_matrix, 255, 0)


            value_sum = 0
            for value in np.nditer(value_matrix):
                if value > 100:
                    value_sum += value
            velocity = value_sum/100
            return value_matrix, velocity









    def sample(self, rate, current_frame_number, frames):
        rate_to_frames = math.ceil(int(self.__fps) / rate)
        if current_frame_number % rate_to_frames != 0:
            return
        self.movement_frames.append(frames[1])

