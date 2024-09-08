from cart_pole_game import CartPoleGame
from bots import NaiveAngleBot
from learners import ActionValueIterationBot
from matplotlib import pyplot as plt
import numpy as np
from human import Human

class LivePlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.im = None
        plt.ion()  # Turn on interactive mode

    def draw(self, utility_map, axis_labels):
        map_1 = utility_map[:, :, 0]

        if self.im is None:
            self.im = self.ax.imshow(map_1)
        else:
            self.im.set_array(map_1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# player = Human()
player = NaiveAngleBot()
# player = ActionValueIterationBot()

# player.set_plotter(LivePlotter())
game = CartPoleGame(player)

game.play_episodes(1)