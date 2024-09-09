import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from cart_pole_player import CartPolePlayer
from cart_pole_game import CartPoleGame

class DiscreteObsSpace:
    def __init__(self, observation_space, action_space, bin_size: list|np.ndarray|int) -> None:
        self.observation_space = observation_space
        self.high: np.ndarray = observation_space.high
        self.low: np.ndarray = observation_space.low

        self.high = np.minimum(self.high, np.full(len(self.high), 4.8))
        self.low = np.maximum(self.low, np.full(len(self.low), -4.8))

        self.bins = np.linspace(self.low, self.high, bin_size+1, axis=1)

        if isinstance(bin_size, int):
            bin_size = [bin_size] * len(observation_space.low)
        
        self.state_map = np.zeros((*bin_size, action_space.n), np.float32)


    def __setitem__(self, coords, setval):
        state_map_idx = tuple(np.digitize(coord, coordbins) for coord, coordbins in zip(coords, self.bins))

        if len(coords) == len(self.state_map.shape):
            state_map_idx = (*state_map_idx, coords[-1])

        self.state_map[state_map_idx] = setval

    def __getitem__(self, coords):
        state_map_idx = tuple(np.digitize(coord, coordbins) for coord, coordbins in zip(coords, self.bins))

        if len(coords) == len(self.state_map.shape):
            state_map_idx = (*state_map_idx, coords[-1])

        return self.state_map[state_map_idx]
    

class ActionValueIterationBot(CartPolePlayer):

    def __init__(self) -> None:
        super().__init__()

        self.explore_probability = 0.99
        self.explore_step = .99/1500
        self.final_explore_probability = 0.0
        self.discount = 0.2
        self.death_penalty = -1.0

        self.obs_state: np.ndarray = None
        
        # state action utility map
        self.sau_map: DiscreteObsSpace = None

        self.state_actions = deque()
        self.last_action: int

        self.time = 0
        self.episode = 0

    def discretize_obs_action_space(self):
        obs_space = self.game.env.observation_space
        act_space = self.game.env.action_space
        self.sau_map = DiscreteObsSpace(obs_space, act_space, 20)

    def set_game(self, game: CartPoleGame):
        self.game = game

    def get_action(self):
        if self.time == 0:
            return random.choice([0, 1])
        
        # explore
        elif random.random() < self.explore_probability:
            action = random.choice([0, 1])

        # exploit
        else:
            action = np.argmax(self.sau_map[self.obs_state])

        self.state_actions.append((*self.obs_state, action))
        
        return action

    def observe(self, observation, reward, terminated, truncated, info):
        self.obs_state = observation

        if self.time == 0:
            pass

        elif terminated:
            last_state_action = self.state_actions.pop()
            self.sau_map[last_state_action] += self.death_penalty
            
            while self.state_actions:
                curr_state_action = self.state_actions.pop()

                self.sau_map[curr_state_action] += self.discount * self.sau_map[last_state_action]

                last_state_action = curr_state_action

            self.episode += 1
            self.time = 0

            self.explore_probability = max(self.explore_probability - self.explore_step, self.final_explore_probability)

        else:
            self.sau_map[self.state_actions[-1]] = 1.0


        self.time += 1

    def draw(self):
        params = {
            'explore_prob': self.explore_probability,
            'time': self.time,
            'episode': self.episode
        }
        return params, self.sau_map.state_map