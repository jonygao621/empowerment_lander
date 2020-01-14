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

from policies import FullPilotPolicy, LaggyPilotPolicy, NoopPilotPolicy, NoisyPilotPolicy, SensorPilotPolicy, CoPilotPolicy

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

def str_of_config(pilot_tol, pilot_type):
  return "{'pilot_type': '%s', 'pilot_tol': %s}" % (pilot_type, pilot_tol)

def run_ep_copilot(policy, env, max_ep_len, pilot, pilot_tol, render=False, pilot_is_human=False):
    obs = env.reset()
    done = False
    totalr = 0.
    trajectory = None
    actions = []
    for step_idx in range(max_ep_len + 1):
        if done:
            trajectory = info['trajectory']
            break
        action = policy.step(obs[None, :], pilot, pilot_tol)
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

    max_ep_len = 1000
    env = LunarLanderEmpowerment(empowerment=0.0, ac_continuous=False)


    full_pilot_policy = FullPilotPolicy(data_dir,
                                        policy_path=os.path.join(data_dir, '01-13-2020 09-46-18/full_pilot_reward.pkl'))
    laggy_pilot_policy = LaggyPilotPolicy(data_dir, full_policy=full_pilot_policy.policy)
    laggy_pilot_policy.step([0,0,0,0,0,0,0,0,0])
    print("Step")
    noisy_pilot_policy = NoisyPilotPolicy(data_dir, full_policy=full_pilot_policy.policy)
    noop_pilot_policy = NoopPilotPolicy(data_dir, full_policy=full_pilot_policy.policy)
    sensor_pilot_policy = SensorPilotPolicy(data_dir, full_policy=full_pilot_policy.policy)

    pilot_names = ['laggy', 'noisy', 'noop', 'sensor']
    pilot_policies = [full_pilot_policy, laggy_pilot_policy, noisy_pilot_policy, noop_pilot_policy]
    configs = []
    pilot_tol_of_id = {
        'noop': 0,
        'laggy': 0.7,
        'noisy': 0.4,
        'sensor': 0
    }


    reward_logs = {}
    copilot_of_training_pilot = {}

    for training_pilot_id, training_pilot_tol in pilot_tol_of_id.items():
        training_pilot_policy = eval('%s_pilot_policy' % training_pilot_id)
        config_kwargs = {
            'pilot_policy': training_pilot_policy,
            'pilot_tol': training_pilot_tol,
            'reuse': True,
            'copilot_scope': 'co_deepq' + training_pilot_id
        }
        print(training_pilot_id)
        co_env = LunarLanderEmpowerment(empowerment=100.0, ac_continuous=False, **config_kwargs)
        copilot_policy = CoPilotPolicy(data_dir)
        copilot_policy.learn(co_env, max_timesteps=max_ep_len, **config_kwargs)
        copilot_of_training_pilot[training_pilot_id] = copilot_policy

    cross_evals={}

    for training_pilot_id, training_pilot_tol in pilot_tol_of_id.items():
        # load pretrained copilot
        training_pilot_policy = eval('%s_pilot_policy' % training_pilot_id)
        config_kwargs = {
            'pilot_policy': training_pilot_policy,
            'pilot_tol': training_pilot_tol,
            'reuse': True
        }

        # evaluate copilot with different pilots
        for eval_pilot_id, eval_pilot_tol in pilot_tol_of_id.items():
            eval_pilot_policy = eval('%s_pilot_policy' % eval_pilot_id)
            copilot_policy = copilot_of_training_pilot[training_pilot_id]

            co_env_eval = LunarLanderEmpowerment(empowerment=100.0, ac_continuous=False, pilot_policy=eval_pilot_policy)
            cross_evals[(training_pilot_id, eval_pilot_id)] = [run_ep_copilot(copilot_policy, co_env_eval, pilot=eval_pilot_policy, pilot_tol=eval_pilot_tol, render=False, max_ep_len=max_ep_len)[:2] for _ in
                                                               range(100)]

