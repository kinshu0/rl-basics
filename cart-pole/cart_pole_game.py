import gymnasium as gym
import pygame
from cart_pole_player import CartPolePlayer


# FPS = 1000
FPS = 10

class CartPoleGame:
    def __init__(self, player: CartPolePlayer) -> None:
        self.env = gym.make('CartPole-v1', render_mode="human")
        observation, info = self.env.reset()

        self.player = player

        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        self.clock = pygame.time.Clock()

        self.running = False
        
    def step(self):
        action = self.player.get_action()
        if action is None:
            return

        observation, reward, terminated, truncated, info = self.env.step(action)
        self.player.observe(observation, reward, terminated, truncated, info)

        if terminated or truncated:
            self.env.reset()
            self.running = False

    def play(self):
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            self.step()

            self.clock.tick(FPS)

        self.env.reset()

    def play_episodes(self, n):
        for i in range(n):
            self.play()

    