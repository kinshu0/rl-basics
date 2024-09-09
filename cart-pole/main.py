from cart_pole_game import CartPoleGame
from cart_pole_player import CartPolePlayer

from matplotlib import pyplot as plt

from bots import NaiveAngleBot
from learners import ActionValueIterationBot
import numpy as np
from human import Human

from matplotlib import animation

class Engine:
    def __init__(self, game: CartPoleGame, player: CartPolePlayer) -> None:
        self.game = game
        self.player = player
        self.clock = None
        self.fps = None

        self.fig, self.aximg = plt.subplots(1, 2)
        self.game_viz = self.aximg[0].imshow(np.zeros((300, 400, 3)), interpolation='none')

        self.player_state_viz = self.aximg[1].imshow(np.zeros((20, 20)), interpolation='none', cmap='Spectral', vmin=-20, vmax=20)


    def draw(self):
        game_viz = self.game.draw()
        self.game_viz.set_data(game_viz)

        utility: np.ndarray = self.player.draw()
        player_viz = utility[:, :, 0]

        self.player_state_viz.set_data(player_viz)
        return self.aximg
    
    def update(self, frame):
        while self.game.update():
            return self.draw()

    def run(self, frame):
        ani = animation.FuncAnimation(fig=self.fig, func=self.update, interval=30, cache_frame_data=False, save_count=2)
        plt.show()
    



# player = Human()
# player = NaiveAngleBot()
player = ActionValueIterationBot()

game = CartPoleGame(player, episodes=100)

engine = Engine(game, player)
engine.run(100)