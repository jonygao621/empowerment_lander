import numpy as np
from baselines import deepq

class NoisyPilotPolicy(object):
    def __init__(self, full_policy_path=None, full_policy=None):
        self.full_policy = full_policy
        if full_policy_path is not None and self.full_policy is None:
            self.full_policy = deepq.deepq.load_act(full_policy_path)
        elif full_policy is None and self.full_policy is None:
            raise NotImplementedError

    def step(self, obs, noise_prob=0.15):
        action = self.full_policy._act(obs)[0]
        if np.random.random() < noise_prob:
            action = (action + 3) % 6 #thuster on / off
            action = action // 3 * 3 + (action + np.random.randint(1, 3)) % 3

        return action