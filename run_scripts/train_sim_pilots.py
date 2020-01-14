from __future__ import division
import os
from envs import LunarLanderEmpowerment, LunarLander

from policies import FullPilotPolicy
from datetime import datetime

if __name__ == '__main__':
    data_dir = os.path.join('data', 'lunarlander-sim')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    cur_datetime = datetime.now().strftime("%m-%d-%Y %H-%M-%S")

    data_dir = os.path.join(data_dir, cur_datetime)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    env = LunarLanderEmpowerment(empowerment=0.0, ac_continuous=False)
    max_ep_len = 1000
    n_training_episodes = 500
    full_pilot_scope = 'full_pilot'
    max_timesteps = max_ep_len * n_training_episodes
    full_pilot_policy = FullPilotPolicy(data_dir)
    full_pilot_policy.learn(env, max_timesteps)
