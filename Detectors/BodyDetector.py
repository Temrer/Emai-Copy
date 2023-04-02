import cv2
import mediapipe as mp
import numpy as np

class BodyDetector():
    def __init__(self, screen_size):
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_drawing_styles = mp.solutions.drawing_styles
        self.__mp_pose           = mp.solutions.pose
        self.__pose_detect = mp.solutions.pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.__height = screen_size[0]
        self.__width  = screen_size[1]

    def process(self, frames, current_frame_number):

        frame = cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.__width//2, self.__height//2))
        results = self.__pose_detect.process(frame)
        output_frame = np.zeros_like(frame)

        self.__mp_drawing.draw_landmarks(
            output_frame,
            results.pose_landmarks,
            self.__mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.__mp_drawing_styles.get_default_pose_landmarks_style())

        output_frame = cv2.resize(output_frame, (self.__width, self.__height))
        cv2.imshow('pose', output_frame)

    def close(self):
        self.__pose_detect.close()
