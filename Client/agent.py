import random
from base_agent import BaseAgent
from game_data import GameData


class Agent(BaseAgent):
    def do_move(self, game_data: GameData):
        return ["N", "S", "E", "W"][random.randint(0, 3)]


if __name__ == "__main__":
    agent = Agent()
    agent.play()
