from Libraries.hand_module import *
import math


class handLib:
    def __init__(self):
        self.hands = []
        self.ids = []
        self.__frame_size = [1,1]
        self.__face_avg = 0
        self.__fh_ratio = 0

    def __generate_id(self, label):
        for idx in self.ids:
            if idx['label'] == label:
                return idx['id'], False

        unique_id = {
            'id': len(self.ids),
            'label' : label
        }
        self.ids.append(unique_id)
        return unique_id['id'], True

    def __calc_velocity(self, pos1, time1, pos2, time2):
        unit =  int(self.__fh_ratio * self.__face_avg)
        abs_dist = math.sqrt( (pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2 )
        dist = abs_dist/unit
        velocity = dist/(time2 - time1)
        return velocity

    def __get_position_form_coords(self, coords):
        return [int(coords.x + (coords.x1 - coords.x) / 2), int(coords.y + (coords.y1 - coords.y) / 2)]

    def __update_hand_history(self, idx, hand):
        self.hands[idx].update_history(hand)

    def __process_landmark(self, landmark):
        x = int(landmark.x * self.__frame_size[0])
        y = int(landmark.y * self.__frame_size[1])
        return [x,y]

    def __compile_hand_struct(self, skeleton, time):
        hand_root = self.__process_landmark(skeleton.landmark[0])
        hand_tip  = self.__process_landmark(skeleton.landmark[12])
        hand_struct = Hand(hand_root, hand_tip, time)
        return hand_struct

    def __compile_hand(self, idx, hand):
        hand_entry = HandEntry(idx, hand)
        return hand_entry


    def __verify_id(self, idx, hand_struct):
        max_velocity = 9

        entry = self.get_hand_history_entry(idx, -1)
        proposed_pos, proposed_time = entry.h_root, entry.time
        if entry.time == 0:
            return idx
        if proposed_time == hand_struct.time:
            proposed_velocity = max_velocity+1
        else:
            proposed_velocity = self.__calc_velocity(proposed_pos, proposed_time,
                                                     hand_struct.h_root, hand_struct.time)

        min_velocity = 999
        min_id = idx
        if proposed_velocity > max_velocity:
            for hand_wrapper in self.hands:
                #still needs work. does not prevent it from switching hands
                #velocity based identification is not reliable
                #position based identification will jump when the hand goes out of frame

                test_id = hand_wrapper.get_id()
                hand = hand_wrapper.get_history_entry(-1)
                test_time = hand.time
                if hand_struct.time - test_time :
                    test_pos = hand.h_root
                    test_velocity = self.__calc_velocity(test_pos, test_time,
                                                         hand_struct.h_root, hand_struct.time)
                    if  test_velocity < min_velocity:
                        min_velocity = test_velocity
                        min_id = test_id

        return min_id









    def update_hand(self, label:str, skeleton, time):
        """
        Function that updates the database of hands
        :param label: the handedness of the hand "Left" || "Right"
        :param skeleton: skeleton of the hand (obtained through mediapipe.solutions.hands)
        :param time: time at which the hand was processed
        :return: None
        """
        idx, generated_new_id = self.__generate_id(label)
        hand_struct = self.__compile_hand_struct(skeleton, time)

        if generated_new_id:
            hand = self.__compile_hand(idx, hand_struct)
            self.hands.insert(idx, hand)
        else:
            idx = self.__verify_id(idx, hand_struct)
            self.__update_hand_history(idx, hand_struct)

    def get_id(self, label:str):
        """
        Gets the id of specified label
        :param label: the handedness of the hand "Left" || "Right"
        :return: id of the specified label
        """
        for idx in self.ids:
            if idx['label'] == label:
                return idx['id']
        print('There is no hand with such label')

    def get_hand_history(self, idx:int):
        """
        Gets the history of the specified id

        :param idx: id of the hand
        can be obtained by calling get_id()
        :return: a list containing locations and timestamps of the hand
        """
        try:
            return self.hands[idx].get_history()
        except IndexError as ie:
            raise ie

    def get_hand_history_entry(self, idx:int, index:int):
        """
        Gets the entry from the history of specified hand
        :param idx: id of the hand
        :param index: the index of specified entry in the hand's history
        """
        try:
            return self.hands[idx].get_history_entry(index)
        except IndexError as ie:
            raise ie

    def get_label_from_id(self, idx):
        return self.ids[idx]['label']

    def set_frame_size(self, size):
        self.__frame_size = size

    def update_frame_variables(self, fh_ratio, face_avg):
        self.__fh_ratio = fh_ratio
        self.__face_avg = face_avg
