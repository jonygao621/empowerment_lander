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
