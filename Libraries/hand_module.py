class Hand:
    def __init__(self, hand_root, hand_tip, time):
        self.h_root = hand_root
        self.h_tip  = hand_tip
        self.time   = time

    def __str__(self):
        return str([self.h_root, self.h_tip, self.time])

class HandEntry:
    def __init__(self, idx, hand):
        self.__id = idx
        self.__hand_history = []
        self.__hand_history.append(hand)

    def get_history(self):
        return self.__hand_history

    def get_history_entry(self, index):
        return self.__hand_history[index]

    def update_history(self, hand):
        self.__hand_history.append(hand)

    def get_id(self):
        return self.__id