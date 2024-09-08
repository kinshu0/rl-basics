from abc import ABC, abstractmethod

'''
observation = cart_position[-4.8, 4.8], cart_velocity[-inf, inf], pole_angle_rad=[-0.418, 0.418], pole_angular_velocity=[-inf, inf]
'''
class CartPolePlayer(ABC):
    PUSH_LEFT = 0
    PUSH_RIGHT = 1

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def observe(self):
        pass
