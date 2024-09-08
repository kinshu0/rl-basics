from cart_pole_player import CartPolePlayer

class NaiveAngleBot(CartPolePlayer):
    def __init__(self) -> None:
        super().__init__()
        self.time_step = 0
        self.last_obs = None
        
    def get_action(self):
        if self.last_obs is None:
            return CartPolePlayer.PUSH_LEFT

        pos, v, ang, ang_v = self.last_obs
        if ang < 0:
            return CartPolePlayer.PUSH_LEFT
        else:
            return CartPolePlayer.PUSH_RIGHT

    def observe(self, observation, reward, terminated, truncated, info):
        self.last_obs = observation
        print(f'{observation=}; {reward=}; {terminated=}; {truncated=}; {info=}')
        self.time_step += 1