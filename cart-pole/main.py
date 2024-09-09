from cart_pole_game import CartPoleGame
from cart_pole_player import CartPolePlayer

from matplotlib import pyplot as plt

from bots import NaiveAngleBot
from learners import ActionValueIterationBot
import numpy as np
from human import Human

from matplotlib import animation
from matplotlib.gridspec import GridSpec

class Engine:
    def __init__(self, game: CartPoleGame, player: CartPolePlayer) -> None:
        self.game = game
        self.player = player
        self.clock = None
        self.fps = None

        self.fig = plt.figure()
        gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

        self.aximg = [self.fig.add_subplot(gs[0]), self.fig.add_subplot(gs[1])]
        self.game_viz = self.aximg[0].imshow(np.zeros((300, 400, 3)), interpolation='none')
        self.aximg[0].set_xticks([])
        self.aximg[0].set_yticks([])
        self.aximg[0].set_title("Game State")

        self.player_state_viz = self.aximg[1].imshow(np.zeros((20, 20)), interpolation='none', cmap='Spectral', vmin=0.0, vmax=1.0)
        self.aximg[1].set_xlabel('Angle θ')
        self.aximg[1].set_ylabel('Angular Velocity ω')
        self.aximg[1].set_title("Player State")

        self.player_params_text = self.fig.text(0.95, 0.05, '', ha='right', va='bottom')

        plt.tight_layout()

    def draw(self):
        game_viz = self.game.draw()
        self.game_viz.set_data(game_viz)

        params, utility = self.player.draw()
        theta_w_mean = np.mean(utility, (0, 1))
        player_viz: np.ndarray = theta_w_mean[:, :, 1] - theta_w_mean[:, :, 0]

        params_text = '\n'.join([f'{k}: {v}' for k, v in params.items()])
        self.player_params_text.set_text(params_text)
        
        player_viz = (player_viz - player_viz.min()) / (player_viz.max() - player_viz.min())
        
        self.player_state_viz.set_data(player_viz)

        return self.aximg
    
    def update(self, frame):
        while self.game.update():
            return self.draw()

    def run(self):
        ani = animation.FuncAnimation(fig=self.fig, func=self.update, interval=0, cache_frame_data=False, save_count=2)
        plt.show()

    def run_no_disp(self):
        while self.game.update():
            pass
    



# player = Human()
# player = NaiveAngleBot()
player = ActionValueIterationBot()

game = CartPoleGame(player, episodes=2000)

engine = Engine(game, player)
engine.run()
# engine.run_no_disp()