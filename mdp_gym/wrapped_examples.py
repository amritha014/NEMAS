"""Examples from emdp.examples wrapped as gym environments"""

# from emdp import examples
from gym_wrap import GymToMDP

class MDP1_v1(GymToMDP):
    def __init__(self):
        super().__init__(mdp1_v1())


