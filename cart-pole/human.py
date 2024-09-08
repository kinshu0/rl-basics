import pygame
from cart_pole_player import CartPolePlayer

class Human(CartPolePlayer):
    def get_action(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            return CartPolePlayer.PUSH_LEFT
        
        elif keys[pygame.K_RIGHT]:
            return CartPolePlayer.PUSH_RIGHT
        
        return None

    def observe(self, observation, reward, terminated, truncated, info):
        pass