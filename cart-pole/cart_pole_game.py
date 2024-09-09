import gymnasium as gym
from cart_pole_player import CartPolePlayer
import matplotlib.pyplot as plt

# FPS = 1000
FPS = 10

class CartPoleGame:
    def __init__(self, player: CartPolePlayer, episodes: int) -> None:
        self.env = gym.make('CartPole-v1', render_mode="rgb_array")
        observation, info = self.env.reset()

        self.player = player
        self.running = False

        self.episodes_remaining = episodes

        self.active = True
        
    def step(self):
        action = self.player.get_action()
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.player.observe(observation, reward, terminated, truncated, info)

        if terminated or truncated:
            self.running = False

    def update(self):
        if self.running:
            self.step()
            
        elif self.episodes_remaining > 0:
            self.episodes_remaining -= 1
            self.env.reset()
            self.running = True

        else:
            self.active = False

        return self.active
        

    def draw(self):
        return self.env.render()
