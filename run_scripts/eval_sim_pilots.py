from __future__ import division
import pickle
import random
import os
import math
import types
import uuid
import time
from copy import copy
from collections import defaultdict, Counter

import numpy as np
import gym
from gym import spaces, wrappers
from gym.envs.registration import register
from envs import LunarLanderEmpowerment, LunarLander

from policies import FullPilotPolicy
from policies import LaggyPilotPolicy

import tensorflow as tf

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.common import models
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.deepq import ActWrapper

from matplotlib import pyplot as plt

from utils.env_utils import *

def run_ep(policy, env, max_ep_len, render=False, pilot_is_human=False):
    obs = env.reset()
    done = False
    totalr = 0.
    trajectory = None
    actions = []
    for step_idx in range(max_ep_len + 1):
        if done:
            trajectory = info['trajectory']
            break
        action = policy.step(obs[None, :])
        obs, r, done, info = env.step(action)
        actions.append(action)
        if render:
            env.render()
        totalr += r
    outcome = r if r % 100 == 0 else 0
    return totalr, outcome, trajectory, actions

if __name__ == '__main__':
    data_dir = os.path.join('data', 'lunarlander-sim')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    max_ep_len = 300000
    env = LunarLanderEmpowerment(empowerment=0.0, ac_continuous=False)

    full_pilot_policy = FullPilotPolicy(data_dir, policy_path= os.path.join(data_dir, 'full_pilot_reward_longtrain.pkl'))
    laggy_pilot_policy = LaggyPilotPolicy(data_dir, full_policy=full_pilot_policy.policy)

    pilot_names = ['full', 'laggy']
    n_eval_eps = 100

    pilot_evals = [
        list(zip(*[run_ep(eval('%s_pilot_policy' % pilot_name), env, render=False, max_ep_len=max_ep_len) for _ in range(n_eval_eps)])) for
        pilot_name in pilot_names]

    mean_rewards = [np.mean(pilot_eval[0]) for pilot_eval in pilot_evals]
    outcome_distrns = [Counter(pilot_eval[1]) for pilot_eval in pilot_evals]

    print('\n'.join([str(x) for x in zip(pilot_names, mean_rewards, outcome_distrns)]))