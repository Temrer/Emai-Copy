import mediapipe as mp
import numpy as np

from Libraries.Hands import hand_lib as hlib
import cv2


class HandDetector:
    def __init__(self, fps, screen_size):
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_drawing_styles = mp.solutions.drawing_styles
        self.__mp_hands = mp.solutions.hands
        self.__mp_face_detection = mp.solutions.face_detection
        self.__hand_lib = hlib.handLib()
        self.__hand_detect = mp.solutions.hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.7,
            max_num_hands=2,
        ).__enter__()
        self.__hand_lib.upload('fps', fps)
        self.__height = screen_size[0]
        self.__width  = screen_size[1]

        #
        self.sample_rate = 15



    def __process_hand(self, results, frames, draw=False, annotations=False):
        roi_list = []

        hand_frame = np.zeros_like(frames[1])

        if not results.multi_hand_landmarks:
            return frames, None, cv2.cvtColor(hand_frame, cv2.COLOR_BGR2GRAY)

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            x = self.__width
            x1 = 0
            y = self.__height
            y1 = 0

            for lm in hand_landmarks.landmark:
                x2, y2 = int(lm.x * self.__width), int(lm.y * self.__height)
                if x2 > x1:
                    x1 = x2
                if x2 < x:
                    x = x2
                if y2 > y1:
                    y1 = y2
                if y2 < y:
                    y = y2

            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x1 > self.__width:
                x1 = int(self.__width)
            if y1 > self.__height:
                y1 = int(self.__height)

            roi = self.__hand_lib.coords(y, y1, x, x1)
            hentry = self.__hand_lib.hand_entry(frames[0], frames[1], roi, 40)
            hsv_frame, avg = self.__hand_lib.calc_velocity(hentry)
            current_roi = hentry.current_roi
            TRESHOLD = 30
            roi_list.append([hsv_frame, avg, current_roi])

            if draw:
                self.__mp_drawing.draw_landmarks(
                    hand_frame,
                    hand_landmarks,
                    self.__mp_hands.HAND_CONNECTIONS,
                    self.__mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.__mp_drawing_styles.get_default_hand_connections_style())
                if annotations:
                    if avg > TRESHOLD:
                        frames[1] = cv2.putText(frames[1], 'moving', (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                                (0, 255, 0))
                    else:
                        frames[1] = cv2.putText(frames[1], 'resting', (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                                (0, 255, 0))

        return frames, roi_list, cv2.cvtColor(hand_frame, cv2.COLOR_BGR2GRAY)

    def process(self, frames, current_frame_number):
        frame = cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB)

        results = self.__hand_detect.process(frame)

        frames, roi_list, hand_frame = self.__process_hand(results, frames, draw=True)
        if roi_list is not None:
            movement = False
            for entry in roi_list:
                if entry[1] > 30:
                    movement = True
            if movement:
                self.__hand_lib.sample(self.sample_rate, current_frame_number, frames)
        return frames

    def close(self):
        self.__hand_detect.close()

    def get_movement_frames(self):
        return  self.__hand_lib.movement_frames

