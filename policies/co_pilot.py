from policies.co_build_graph import *
import tensorflow as tf
import baselines.common.tf_util as U
import tempfile

import os
import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func

from policies.co_build_graph import *
from utils.env_utils import *

import uuid

def learn(
        env,
        network,
        seed=None,
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=1,
        batch_size=32,
        print_freq=100,
        checkpoint_freq=10000,
        checkpoint_path=None,
        learning_starts=1000,
        gamma=1.0,
        target_network_update_freq=500,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        param_noise=False,
        num_cpu=5,
        callback=None,
        scope='deepq',
        pilot_tol=0,
        pilot_is_human=False,
        reuse=False,
        load_path=None,
        **network_kwargs):
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    if sess is None:
        sess = U.make_session(num_cpu=num_cpu)
        sess.__enter__()

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    using_control_sharing = pilot_tol > 0

    act, train, update_target, debug = co_build_train(
        scope=scope,
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        reuse=reuse,
        using_control_sharing=using_control_sharing
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = deepq.ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    episode_outcomes = []
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    prev_t = 0
    rollouts = []

    if not using_control_sharing:
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                     initial_p=1.0,
                                     final_p=exploration_final_eps)

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        for t in range(total_timesteps):
            masked_obs = mask_helipad(obs)

            act_kwargs = {}
            if using_control_sharing:
                act_kwargs['pilot_action'] = env.unwrapped.pilot_policy(obs[None, :9])
                act_kwargs['pilot_tol'] = pilot_tol
            else:
                act_kwargs['update_eps'] = exploration.value(t)

            #action = act(masked_obs[None, :], **act_kwargs)[0][0]
            action = act(np.array(masked_obs)[None], **act_kwargs)[0]
            env_action = action
            reset = False
            new_obs, rew, done, info = env.step(env_action)
            # Store transition in the replay buffer.
            masked_new_obs = mask_helipad(new_obs)
            replay_buffer.add(masked_obs, action, rew, masked_new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)

                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            episode_outcomes.append(rew)
            episode_rewards.append(0.0)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            mean_100ep_succ = round(np.mean([1 if x == 100 else 0 for x in episode_outcomes[-101:-1]]), 2)
            mean_100ep_crash = round(np.mean([1 if x == -100 else 0 for x in episode_outcomes[-101:-1]]), 2)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("mean 100 episode succ", mean_100ep_succ)
                logger.record_tabular("mean 100 episode crash", mean_100ep_crash)
                logger.dump_tabular()

            if checkpoint_freq is not None and t > learning_starts and num_episodes > 100 and t % checkpoint_freq == 0 and (
                    saved_mean_reward is None or mean_100ep_reward > saved_mean_reward):
                if print_freq is not None:
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(
                        saved_mean_reward, mean_100ep_reward))
                save_variables(model_file)
                model_saved = True
                saved_mean_reward = mean_100ep_reward

        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    reward_data = {
        'rewards': episode_rewards,
        'outcomes': episode_outcomes
    }

    return act, reward_data

class CoPilotPolicy(object):
    def __init__(self, data_dir, policy_path=None):
        self.policy = None
        self.policy_path = policy_path
        self.data_dir = data_dir
        if policy_path is not None:
            self.policy = deepq.deepq.load_act(policy_path)

    def learn(self, env, max_timesteps, copilot_scope='co_deepq', pilot_tol=0, reuse=False):

        if copilot_scope is not None:
            scope = copilot_scope
        elif copilot_scope is None:
            scope = str(uuid.uuid4())

        self.policy = learn(
            env,
            scope=scope,
            network='mlp',
            total_timesteps=max_timesteps,
            pilot_tol=pilot_tol,
            reuse=reuse,
            lr=1e-3,
            target_network_update_freq=500,
            gamma=0.99
        )

        self.policy_path = os.path.join(self.data_dir, 'full_pilot_reward_longtrain.pkl')
        self.policy.save_act(path=self.policy_path)

    def step(self, observation, **kwargs):
        return self.policy.step(observation, **kwargs)

