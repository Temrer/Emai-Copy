import cv2
import hand_lib as hlib
import numpy as np
from time import time
from helper import *
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
hand_lib = hlib.handLib()




def process_face(results, frame, width, height, draw = False):
    if not results:
        print('No face detected')
        return None, frame
    relx = results.detections[0].location_data.relative_bounding_box.xmin
    rely = results.detections[0].location_data.relative_bounding_box.ymin
    relx1 = results.detections[0].location_data.relative_bounding_box.width
    rely1 = results.detections[0].location_data.relative_bounding_box.height
    x = int(width * relx)
    x1 = int(width * (relx + relx1))
    y = int(height * rely)
    y1 = int(height * (rely + rely1))

    face_img = frame[y:y1, x:x1]
    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

    if draw:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)
    return face_img, frame

def get_hand_label(index, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            label = classification.classification[0].label
            output = label
    return output


def process_hand(results, frame, width, height, frame_count, original_fps, draw = False, annotations = False):
    hand_frames = []

    if not results.multi_hand_landmarks:
        return [], frame

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        x = width
        x1 = 0
        y = height
        y1 = 0

        for lm in hand_landmarks.landmark:
            x2, y2 = int(lm.x * width), int(lm.y * height)
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
        if x1 > width:
            x1 = int(width)
        if y1 > height:
            y1 = int(height)


        label = str(get_hand_label(idx, results))
        if label != 'None':
            hand_lib.update_hand(label, hlib.coords(x, x1, y, y1), hand_landmarks, calc_time(original_fps, frame_count))
            hand_frames.append(cv2.cvtColor(frame[y:y1, x:x1], cv2.COLOR_RGB2BGR))

        if draw:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            if annotations and label != 'None':
                cv2.putText(frame, label, (x, y1+30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1)

    return hand_frames, frame

def prep_first_frame(capture, hands, face_detection, frame_count, original_fps):
    cap = capture
    ret, frame = cap.read()

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cv2.imshow('first frame', frame)
    cv2.waitKey(0)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #  processing hand landmarks
    results = hands.process(frame)
    hand_frames, _frame = process_hand(results, frame, width, height, frame_count, original_fps)

    #  processing face
    results = face_detection.process(frame)
    face_img, _frame = process_face(results, frame, width, height)


    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


    #  getting average size of hands by the first diagonal
    head_diag = calc_diag(face_img)
    #       calculating average size of hands
    hand_diags = 0
    hands = 0
    for hand in hand_frames:
        hands += 1
        hand_diags += calc_diag(hand)
    hand_avg_diag = hand_diags / hands

    fh_ratio = head_diag / hand_avg_diag  # face-head ratio
    # will be used to calculate relative velocity of hand

    cv2.imshow('MediaPipe Hands', frame)
    cv2.waitKey(0)
    return fh_ratio, hand_avg_diag



def main():
    print('start')
    start_time = 1
    video = 'HandsMotion1.mp4'
    process_frequency = 30


    cap = cv2.VideoCapture(video)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    hand_detect = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.8
    ).__enter__()
    face_detect = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ).__enter__()

    fh_ratio, h_avg_diag = prep_first_frame(cap, hand_detect, face_detect, 0, original_fps)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    converted_freq = 1/process_frequency
    #  marking the last time the frame has been processed
    hand_frames = []
    hand_ids = []
    last_tick = frame_count
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        #  if enough time has elapsed after last processed frame
        if (frame_count - last_tick) >= int(process_frequency/original_fps):
            last_tick = frame_count
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hand_detect.process(frame)
            hand_frames, processed_frame = process_hand(
                results,
                frame,
                width,
                height,
                frame_count,
                original_fps,
                draw=True,
                annotations=True
            )

            # calculate the velocity of the right hand
            try:
                old_pos, old_time = hand_lib.decompile_history_entry(hand_lib.get_hand_history_entry(0, -2))
                new_pos, new_time = hand_lib.decompile_history_entry(hand_lib.get_hand_history_entry(0, -1))
                velocity = calc_velocity(old_pos, old_time, new_pos, new_time, fh_ratio, h_avg_diag)
                if velocity > 1:
                    frame.flags.writeable = True
                    frame = cv2.line(frame, tuple(old_pos), tuple(new_pos), (0, 100, 255), 10)
                    # frame.flags.writeable = False
            except IndexError as ie:
                pass

            #  Draw the hand annotations on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break


    hand_detect.close()
    face_detect.close()
    cap.release()




if __name__ == '__main__':
    main()
