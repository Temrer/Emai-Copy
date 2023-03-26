import cv2
from Libraries import hand_lib as hlib
from Libraries.helper import *
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
hand_lib = hlib.handLib()
hand_detect = None
face_detect = None
width  = None
height = None
helper = Helper()



def process_face(results, frame, draw = False):
    global width, height
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


def process_hand(results, frames, draw = False):
    global height, width
    roi_list = []

    if not results.multi_hand_landmarks:
        return frames, None

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

        roi = hand_lib.coords(y, y1, x, x1)
        hentry = hand_lib.hand_entry(frames[0], frames[1], roi, 40)
        hsv_frame, avg = hand_lib.calc_velocity(hentry)
        current_roi = hentry.current_roi
        TRESHOLD = 30
        if avg > TRESHOLD:
            current_roi = cv2.putText(current_roi, 'moving', (0,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
        else:
            current_roi = cv2.putText(current_roi, 'resting', (0,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
        roi_list.append([hsv_frame, avg, current_roi])





        if draw:
            mp_drawing.draw_landmarks(
                frames[1],
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    return frames, roi_list

def main():
    print('start')
    video = 'HandsMotion1.mp4'
    process_frequency = 2
    camera = False


    if camera:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        result = cv2.VideoWriter('Output.mp4', fourcc, 30, (1280, 960))
        original_fps = 30
        helper.set_prop('fps', original_fps)
    else:
        cap = cv2.VideoCapture(video)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        helper.set_prop('fps', original_fps)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    frame_count = 0
    global hand_detect, face_detect
    hand_detect = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.7,
        max_num_hands=2,
    ).__enter__()
    face_detect = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ).__enter__()


    global width, height
    frames = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    width  = int(width//2)
    height = int(height//2)
    frame = cv2.resize(frame, (width, height))
    frames.insert(0, frame)

    ret, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    frames.insert(1, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detect.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frames, roi_list = process_hand(results, frames, draw=True)
    cv2.imshow('MediaPipe Hands', frame)
    cv2.waitKey(0)

    if camera:
        continue_condition = True
    else:
        continue_condition = cap.isOpened()
    while continue_condition:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = cv2.resize(frame, (width, height))
        frames[0] = frames[1]
        frames[1] = frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detect.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames, roi_list = process_hand(results, frames, draw=True)
        if roi_list is not None:
            for index, value in enumerate(roi_list):
                cv2.imshow(str(index), value[2])

        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break



    hand_detect.close()
    face_detect.close()
    cap.release()




if __name__ == '__main__':
    main()
