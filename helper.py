import math
import hand_lib
"""
a library with some methods to aid for the main program
"""


def calc_diag(frame):
    """
    A function that calculates the diagonal of a frame

    :param frame:  image
    :return: float, size of diagonal
    """
    y, x = frame.shape[0], frame.shape[1]
    diag = math.sqrt(y**2 + x**2)
    return diag

def calc_velocity(pos1, time1, pos2, time2, fh_ratio, face_diag):
    unit =  int(fh_ratio * face_diag)
    abs_dist = math.sqrt( (pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2 )
    dist = abs_dist/unit
    velocity = dist/(time2 - time1)
    return velocity

def calc_time(original_fps, current_frame):
    return current_frame/original_fps

def verify_label(coords: hand_lib.coords, frame_count, label, hlib: hand_lib.handLib, max_velocity ):
    pos = [coords.x + int((coords.x1 - coords.x)/2),
           coords.y + int((coords.y1 - coords.y)/2)]

    # we assume that this is the wrong hand
    # we calculate the hand that is the closest from current pos
    closest_dist = 9999999
    closest_id = None
    for hand in hlib.hands:
        virtual_pos = hand['position_history'][-1]['position']

        distance = math.sqrt((virtual_pos[0] - pos[0])**2 + (virtual_pos[1] - pos[1])**2)
        if distance < closest_dist:
            closest_dist = distance
            closest_id = hand['id']

        