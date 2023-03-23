import math

"""
a library with some methods to aid for the main program
"""

class Helper:
    def __init__(self):
        self.__face_avg = None
        self.__hand_avg = None
        self.__fh_ratio = None
        self.__fps      = None
        self.__hand_lib = None

        # identifiers
        self.__face_avg_id = 'face_avg'
        self.__hand_avg_id = 'hand_avg'
        self.__fh_ratio_id = 'fh_ratio'
        self.__fps_id      = 'fps'
        self.__hand_lib_id = 'hand_lib'

    def set_prop(self, prop:str, value):
        """
        sets a property to a specified value

        Properties = ['face_avg','hand_avg','fh_ratio','fps','hand_lib']
        :param prop: the property to be modified
        :param value: the value to be assigned
        :return: None
        """

        if prop == self.__face_avg_id:
            self.__face_avg = value

        elif prop == self.__hand_avg_id:
            self.__hand_avg = value

        elif prop == self.__fh_ratio_id:
            self.__fh_ratio = value

        elif prop == self.__fps_id:
            self.__fps = value

        elif prop == self.__hand_lib_id:
            self.__hand_lib = value

        else:
            raise ValueError("No such property exists")

    def __calc_diag(self, x, y):
        return math.sqrt(y**2 + x**2)

    def calc_diag_from_frame(self, frame):
        """
        A function that calculates the diagonal of a frame

        :param frame:  image
        :return: float, size of diagonal
        """
        y, x = frame.shape[0], frame.shape[1]
        diag = self.__calc_diag(x,y)
        return diag

    def calc_velocity(self, pos1, time1, pos2, time2):
        unit =  int(self.__fh_ratio * self.__face_avg)
        abs_dist = math.sqrt( (pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2 )
        dist = abs_dist/unit
        velocity = dist/(time2 - time1)
        return velocity

    def calc_time(self, current_frame):
        return current_frame/self.__fps

    def verify_label(self, coords, frame_count, label, max_velocity):


        # we assume that this is the wrong hand
        # we calculate the hand that is the closest from current pos
        closest_dist = 9999999
        closest_id = None
        for id in self.__hand_lib.hands:
            hand = self.__hand_lib.hands[id]
            #virtual_pos = hand['position_history'][-1]['position']

            # distance = math.sqrt((virtual_pos[0] - pos[0])**2 + (virtual_pos[1] - pos[1])**2)
            # if distance < closest_dist:
            #     closest_dist = distance
            #     closest_id = hand['id']

        if not closest_id:
            return False, 0
        return True, int(closest_id)

