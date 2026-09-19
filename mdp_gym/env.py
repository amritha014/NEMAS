"""
A simple grid world environment
"""

import numpy as np
import random
from mdp_gym.common import MDP
from mdp_gym.exceptions import EpisodeDoneError, InvalidActionError
from mdp_gym.actions import right_action, top_action, left_action, bottom_action
from mdp_gym.helper_utilities import flatten_state, unflatten_state

class GridWorldMDP(MDP):
    def __init__(self, P, R, discount, epsilon, max_iter, skip_check=False):
        """
        (!) if terminal_states is not empty then there will be an absorbing state. So
            the actual number of states will be size x size + 1
            if there is a terminal state, it should be the last one.
        :param P: Transition matrix |S| x |A| x |S|
        :param R: Transition matrix |S| x |A|
        :param gamma: discount factor
        """
        # if not convert_terminal_states_to_ints:
        #     terminal_states = list(map(lambda tupl: int(size * tupl[0] + tupl[1]), terminal_states))
        self.size =  size
        self.human_state = (None, None)
        self.has_absorbing_state = len(terminal_states) > 0
        super().__init__(P, R, gamma, p0, terminal_states, seed=seed, skip_check=skip_check)

    def reset(self):
        super().reset()
        self.human_state = self.unflatten_state(self.current_state)
        return self.current_state

    def flatten_state(self, state):
        """Flatten state (x,y) into a one hot vector"""
        return flatten_state(state, self.size, self.state_space)

    def unflatten_state(self, onehot):
        """Unflatten a one hot vector into a (x,y) pair"""
        return unflatten_state(onehot, self.size, self.has_absorbing_state)

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.human_state = self.unflatten_state(self.current_state)
        return state, reward, done, info

    def set_current_state_to(self, tuple_state):
        return super().set_current_state_to(self.flatten_state(tuple_state).argmax())
