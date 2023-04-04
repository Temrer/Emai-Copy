import cv2
from Libraries.helper import *
import Detectors.HandDetector as HandDetector
import Detectors.BodyDetector as BodyDetector
from time import sleep, time
from os.path import join
import os


helper = Helper()





def main():
    print('start')
    video = 'HandsMotion1.mp4'
    process_frequency = 2
    camera = False
    sample_rate = 10


    if camera:
        width = 1280
        height = 960
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # result = cv2.VideoWriter('Output.mp4', fourcc, 30, (width, height))
        original_fps = 30
        helper.set_prop('fps', original_fps)
        _ret, _frame = cap.read()
    else:
        cap = cv2.VideoCapture(video)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        helper.set_prop('fps', original_fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    frame_count = 0

    frames = []

    ret, frame = cap.read()
    width  = int(width//2)
    height = int(height//2)
    hand_detect = HandDetector.HandDetector(original_fps, (height, width))
    body_detect = BodyDetector.BodyDetectorCPU((height, width), original_fps)
    print(ret)
    old_frame = cv2.resize(frame, (width, height))
    frames.insert(0, old_frame)

    ret, frame = cap.read()
    new_frame = cv2.resize(frame, (width, height))
    frames.insert(1, new_frame)
    frame_count = 2
    # hand_detect.process(frames, frame_count)
    body_detect.process(frames, frame_count)
    cv2.imshow('MediaPipe Hands', frames[1])

    if camera:
        continue_condition = True
    else:
        continue_condition = cap.isOpened()
    while continue_condition:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        new_frame = cv2.resize(frame, (width, height))
        old_frame = frames[1]
        frames[0] = old_frame
        frames[1] = new_frame
        # frames = hand_detect.process(frames, frame_count)
        velocity, movement = body_detect.process(frames, frame_count)
        if movement:
            body_detect.sample(sample_rate, frame_count, frames)

        cv2.imshow('MediaPipe Hands', frames[1])
        if cv2.waitKey(5) & 0xFF == 27:
            break



    hand_detect.close()
    cap.release()


    path = join(r"J:\Petru\Projects", r"Results\Vid0")
    if not os.path.exists(path):
        os.mkdir(r"J:\Petru\Projects\Results")
        os.mkdir(path)
    os.chdir(path)
    file_name = "Frame"
    for index, frame in enumerate(body_detect.movement_frames):
        cv2.imshow("movement", frame)
        final_file_name = file_name+str(index)+'.jpg'
        cv2.imwrite(final_file_name, frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break
        sleep(0.3)




if __name__ == '__main__':
    main()
