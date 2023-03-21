class coords:
    def __init__(self, x, x1, y, y1):
        self.x = x
        self.x1 = x1
        self.y = y
        self.y1 = y1

class handLib:
    def __init__(self):
        self.hands = {}
        self.ids = []

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

    def __get_position_form_coords(self, coords):
        return [int(coords.x + (coords.x1 - coords.x) / 2), int(coords.y + (coords.y1 - coords.y) / 2)]

    def __update_hand_history(self, idx, coords, time):
        self.hands[idx]['position_history'].append(
            {
                'position': self.__get_position_form_coords(coords),
                'time' : time,
            }
        )


    def __compile_hand(self, idx, coords, skeleton, time):
        hand = {
            'id': idx,
            'coords': {
                'x': coords.x,
                'y': coords.y,
                'x1': coords.x1,
                'y1': coords.y1,
                'width': coords.x1 - coords.x,
                'height': coords.y1 - coords.y,
            },
            'skeleton': skeleton,
            'position_history': [{
                'position': self.__get_position_form_coords(coords),
                'time': time,
            }]
        }
        return hand


    def update_hand(self, label:str, coords:coords, skeleton, time):
        """
        Function that updates the database of hands
        :param label: the handedness of the hand "Left" || "Right"
        :param coords: coordinates of the hand in the image
        formatted as (x, x1, y, y1)
        :param skeleton: skeleton of the hand (obtained through mediapipe.solutions.hands)
        :param time: time at which the hand was processed
        :return: None
        """
        idx, generated_new_id = self.__generate_id(label)
        if generated_new_id:
            hand = self.__compile_hand(idx, coords, skeleton, time)
            self.hands[idx] = hand
        else:
            self.__update_hand_history(idx, coords, time)

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
            return self.hands[idx]['position_history']
        except IndexError as ie:
            raise ie

    def get_hand_history_entry(self, idx:int, index:int):
        """
        Gets the entry from the history of specified hand
        :param idx: id of the hand
        :param index: the index of specified entry in the hand's history
        """
        try:
            return self.hands[idx]['position_history'][index]
        except IndexError as ie:
            raise ie

    def decompile_history_entry(self, entry:dict):
        """
        Transforms the entry into objects
        :param entry: the entry to be decompiled
        :return: the position and time from the entry
        """
        position = entry['position']
        time = entry['time']
        return position, time

    def __foolproof(self, entry):
        pass
