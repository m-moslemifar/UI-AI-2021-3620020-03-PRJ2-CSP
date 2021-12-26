import operator
import numpy as np

EMPTY_CONST = 1000


class GameData:
    def __init__(self, melody: list, agent_pos: tuple, matrix: list, max_turns: int):
        self.__melody        = melody
        self.matrix          = matrix
        self.grid_width      = len(self.matrix)
        self.grid_height     = len(self.matrix[0])
        self.agent_pos       = agent_pos
        self.remaining_turns = max_turns
        self.melody_length   = len(self.__melody)

    def h(self, state: list):
        state1 = [ord(i) - ord("a") + 1 for i in state]
        if len(state) < len(self.__melody):
            for i in range(len(self.__melody) - len(state)):
                state1.append(EMPTY_CONST)
        state2 = [ord(i) - ord("a") + 1 for i in self.__melody]
        return np.linalg.norm(list(map(operator.sub, state1, state2)))
