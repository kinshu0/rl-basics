import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from cart_pole_player import CartPolePlayer

class ActionValueIterationBot(CartPolePlayer):

    def __init__(self) -> None:
        super().__init__()

        # binning angle
        self.theta_lb = -0.418
        self.theta_ub = 0.418
        self.theta_bins = 20
        self.theta_bin_size = (self.theta_ub - self.theta_lb) / self.theta_bins

        # binning angular velocity
        self.w_lb = -0.418
        self.w_ub = 0.418
        self.w_bins = 20
        self.w_bin_size = (self.w_ub - self.w_lb) / self.w_bins

        # push left and push right
        self.num_actions = 2

        # explore probability
        self.explore_p = 0.90

        self.utility: np.ndarray = np.zeros((self.theta_bins+2, self.w_bins+2, self.num_actions), np.float32)

        self.state_actions = deque()


        self.theta_bin: int
        self.w_bin: int
        self.last_action: int

        self.discount = 0.2
        self.death_penalty = -10.0

        self.time = 0
        self.episode = 0

    def get_action(self):
        if self.time == 0:
            return random.choice([0, 1])
        
        # explore
        elif random.random() < self.explore_p:
            action = random.choice([0, 1])

        # exploit
        else:
            action = np.argmax(self.utility[self.theta_bin, self.w_bin])

        self.state_actions.append((self.theta_bin, self.w_bin, action))
        
        return action
    
    def update_state(self, obs):
        pos, v, theta, w = obs
        theta_bin = int((theta - self.theta_lb) / self.theta_bin_size)
        theta_bin = min(theta_bin, self.theta_bins - 1)
        theta_bin = max(theta_bin, 0)

        w_bin = int((w - self.w_lb) / self.w_bin_size)
        w_bin = min(w_bin, self.w_bins - 1)
        w_bin = max(w_bin, 0)

        self.theta_bin = theta_bin
        self.w_bin = w_bin

    def observe(self, observation, reward, terminated, truncated, info):
        self.update_state(observation)

        if self.time == 0:
            pass

        elif terminated:
            last_state_action = self.state_actions.pop()
            self.utility[last_state_action] += self.death_penalty
            
            while self.state_actions:
                curr_state_action = self.state_actions.pop()

                self.utility[curr_state_action] += self.discount * self.utility[last_state_action]

                last_state_action = curr_state_action

            self.episode += 1
            self.time = 0

        else:
            self.utility[self.state_actions[-1]] = 1.0

        self.explore_p = max(self.explore_p - 0.01, .20)

        self.time += 1

    def draw(self):
        return self.utility