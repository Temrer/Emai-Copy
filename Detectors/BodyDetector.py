import cv2
import mediapipe as mp
import numpy as np
from Libraries.helper import Coords
import math

class BodyDetectorCPU():
    def __init__(self, screen_size, fps):
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_drawing_styles = mp.solutions.drawing_styles
        self.__mp_pose           = mp.solutions.pose
        self.__pose_detect = mp.solutions.pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.__fps = fps

        self.__height = screen_size[0]
        self.__width  = screen_size[1]
        self.__width_halved = self.__width//2
        self.__height_halved = self.__height//2
        self.__last_frame = None

        self.MOVEMENT_THRESHOLD = 0.9
        self.movement_frames = []

    def __process_body(self, landmarks_frame, Roi):
        if self.__last_frame is not None:
            hsv = np.zeros(self.__last_frame.shape)
            landmarks_frame_gray = cv2.cvtColor(landmarks_frame, cv2.COLOR_RGB2GRAY)
            landmarks_frame_gray = cv2.resize(landmarks_frame_gray, (self.__width_halved, self.__height_halved))
            landmarks_frame_gray_cut = landmarks_frame_gray[Roi.y1:Roi.y2, Roi.x1:Roi.x2]
            last_landmarks_frame = self.__last_frame[Roi.y1:Roi.y2, Roi.x1:Roi.x2]
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    last_landmarks_frame, landmarks_frame_gray_cut, None, 0.5, 5, 14, 3, 7, 1.5, 0,
                )
            except:
                print(Roi.x1,Roi.x2,Roi.y1,Roi.y2)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
            hsv = cv2.normalize(
                magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX, -1
            )
            hsv_8u = np.uint8(hsv * 255.0)
            avg = 0
            for x in np.nditer(hsv_8u):
                if x > 100:
                    avg += x
            velocity = avg/(self.__width_halved * self.__height_halved)
            if velocity > self.MOVEMENT_THRESHOLD:
                cv2.imshow('curr frame', landmarks_frame_gray)
            self.__last_frame = cv2.cvtColor(landmarks_frame, cv2.COLOR_RGB2GRAY)
            return velocity, velocity > self.MOVEMENT_THRESHOLD
        else:
            self.__last_frame = cv2.cvtColor(landmarks_frame, cv2.COLOR_RGB2GRAY)
            return 0, False

    def process(self, frames, current_frame_number):

        frame = cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.__width_halved, self.__height_halved))
        results = self.__pose_detect.process(frame)
        output_frame = np.zeros_like(frame)

        self.__mp_drawing.draw_landmarks(
            output_frame,
            results.pose_landmarks,
            self.__mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.__mp_drawing_styles.get_default_pose_landmarks_style())
        if results.pose_landmarks is None:
            return 0, 0

        minx = 1
        maxx = 0
        miny = 1
        maxy = 0
        for lm in results.pose_landmarks.landmark:
            if lm.x < minx:
                minx = lm.x
            if lm.y < miny:
                miny = lm.y
            if lm.x > maxx:
                maxx = lm.x
            if lm.y > maxy:
                maxy = lm.y

        if minx < 0:
            minx = 0
        if miny < 0:
            miny = 0
        if maxx < 1:
            maxx = 1
        if maxy < 1:
            maxy = 1

        minx = int(minx * self.__width_halved)
        maxx = int(maxx * self.__width_halved)
        miny = int(miny * self.__height_halved)
        maxy = int(maxy * self.__height_halved)

        Roi = Coords(miny, maxy, minx, maxx)

        velocity, movement = self.__process_body(output_frame, Roi)
        return velocity, movement

    def close(self):
        self.__pose_detect.close()

    def sample(self, rate, current_frame_number, frames):
        rate_to_frames = math.ceil(int(self.__fps) / rate)
        if current_frame_number % rate_to_frames != 0:
            return
        self.movement_frames.append(frames[1])