import cv2
from Libraries.helper import *
import Detectors.HandDetector as HandDetector
import Detectors.BodyDetector as BodyDetector
from time import sleep, time
from os.path import join
import os


helper = Helper()





def main():
    global video_length
    print('start')
    os.chdir(join(os.path.dirname(os.path.dirname(__file__)), "Sources"))
    video = 'Vid3.mp4'
    process_frequency = 2
    camera = False
    sample_rate = 10
    record = True
    base_path = r"J:\Petru\Projects"
    path = join(r"J:\Petru\Projects", r"Results\Vid5")

    if camera:
        units = "seconds"
    else:
        units = "frames"


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
        video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)


    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    frame_count = 0

    frames = []

    ret, frame = cap.read()
    ### Temporary Changes
    # width  = int(width//2)
    # height = int(height//2)

    ratio = width/320
    width = int(width/ratio)
    height = int(height/ratio)


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
        if body_detect.sample_time(sample_rate, frame_count, frames):
            velocity, movement = body_detect.process(frames, frame_count)
            if movement:
                body_detect.sample(sample_rate, frame_count, frames)

        if frame_count % (video_length//1000) == 0:
            print(frame_count / (video_length//1000)/10)

        cv2.imshow('MediaPipe Hands', frames[1])
        if cv2.waitKey(5) & 0xFF == 27:
            break



    hand_detect.close()
    cap.release()

    if record:
        if not os.path.exists(join(base_path, "Results")):
            os.mkdir(join(base_path, "Results"))

        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)
        file_name = "Frame"
        for index, frame in enumerate(body_detect.movement_frames):
            cv2.imshow("movement", frame)
            final_file_name = file_name+str(index)+'.jpg'
            cv2.imwrite(final_file_name, frame)

        print("done")




if __name__ == '__main__':
    main()
