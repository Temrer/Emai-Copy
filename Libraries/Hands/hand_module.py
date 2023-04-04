import cv2
import numpy as np


class Hand:
    def __init__(self, hand_root, hand_tip, time):
        self.h_root = hand_root
        self.h_tip  = hand_tip
        self.time   = time


class HandEntry:
    def __init__(self, last_frame, current_frame, hand_roi, border):
        self.roi = self.__calc_roi(hand_roi, border, (last_frame.shape[0], last_frame.shape[1]))
        self.current_roi = current_frame[
                           self.roi.y1 : self.roi.y2,
                           self.roi.x1 : self.roi.x2
                           ]
        self.last_roi = last_frame[
                        self.roi.y1 : self.roi.y2,
                        self.roi.x1 : self.roi.x2
                        ]


    def get_grayscale_roi(self):
        gray_current_roi = cv2.cvtColor(self.current_roi, cv2.COLOR_BGR2GRAY)
        gray_last_roi    = cv2.cvtColor(self.last_roi, cv2.COLOR_BGR2GRAY)
        return gray_last_roi, gray_current_roi


    def __calc_roi(self, hand_roi, border, frame_size):
        roi_shape  = hand_roi.shape
        roi_width  = roi_shape[1]
        roi_height = roi_shape[0]

        # the outputted roi will be of ratio 1:1
        if roi_width > roi_height:
            edge_len = roi_width +  2 * border
        else:
            edge_len = roi_height +  2 * border

        height_border = (edge_len - roi_height)//2
        width_border  = (edge_len - roi_width )//2
        height_offset = 0
        width_offset  = 0
        if hand_roi.y1 - height_border < 0:
            height_offset = height_border - hand_roi.y1
        elif hand_roi.y2 + height_border > frame_size[0]:
            height_offset = frame_size[0] - hand_roi.y2 - height_border
        if hand_roi.x1 - width_border < 0:
            width_offset = width_border - hand_roi.x1
        elif hand_roi.x2 + width_border > frame_size[1]:
            width_offset =  frame_size[1] - hand_roi.x2 - width_border

        roi_x1 = hand_roi.x1 - width_border + width_offset
        roi_x2 = hand_roi.x2 + width_border + width_offset
        roi_y1 = hand_roi.y1 - height_border + height_offset
        roi_y2 = hand_roi.y2 + height_border + height_offset

        roi = Coords(roi_y1, roi_y2, roi_x1, roi_x2)
        return roi




