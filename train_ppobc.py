import os
from datetime import datetime

import numpy as np
import scipy.signal
import tensorflow as tf
import matplotlib.pyplot as plt

from multiprocessing import Process, Pipe

"""
## Functions and class
"""


def tf_get_mini_batches_gpu(obs_buf, act_buf, adv_buf, ret_buf, logp_buf, batch_size):
    # Use Dataset API to batch and prefetch
    dataset = tf.data.Dataset.from_tensor_slices((
        obs_buf,
        act_buf,
        adv_buf,
        ret_buf,
        logp_buf,
    ))
    dataset = dataset.shuffle(buffer_size=obs_buf.shape[0])
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates (计算折扣累计和，用于计算奖励到期和优势估计)
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:  # GPU-optimized Buffer
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95, last_value=0):
        self.observation_buffer = tf.Variable(tf.zeros((size, observation_dimensions), dtype=tf.float32))
        self.action_buffer = tf.Variable(tf.zeros(size, dtype=tf.int32))
        self.advantage_buffer = tf.Variable(tf.zeros(size, dtype=tf.float32))
        self.reward_env_buffer = tf.Variable(tf.zeros(size, dtype=tf.float32))
        self.return_env_buffer = tf.Variable(tf.zeros(size, dtype=tf.float32))
        self.last_value_tensor = tf.constant([last_value], dtype=tf.float32)
        self.value_buffer = tf.Variable(tf.zeros(size, dtype=tf.float32))
        self.logprobability_buffer = tf.Variable(tf.zeros(size, dtype=tf.float32))
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation_AI, action_AI_agent, reward_env, value, logprobability):
        # Convert tensors to float32 if needed to prevent dtype mismatch
        observation_AI = tf.cast(observation_AI, dtype=tf.float32)
        value = tf.cast(value, dtype=tf.float32)
        logprobability = tf.cast(logprobability, dtype=tf.float32)

        # Store the values in the buffer
        self.observation_buffer[self.pointer].assign(observation_AI[0])  # Store flattened observation
        self.action_buffer[self.pointer].assign(action_AI_agent)
        self.reward_env_buffer[self.pointer].assign(reward_env)
        self.value_buffer[self.pointer].assign(value)
        self.logprobability_buffer[self.pointer].assign(logprobability)
        self.pointer += 1

    def finish_trajectory(self):
        path_slice = slice(self.trajectory_start_index, self.pointer)

        # Slice relevant parts of the buffers
        rewards = self.reward_env_buffer[path_slice]
        values = self.value_buffer[path_slice]

        # Append `last_value` using tf.concat
        rewards = tf.concat([rewards, self.last_value_tensor], axis=0)
        values = tf.concat([values, self.last_value_tensor], axis=0)

        # Calculate deltas using TensorFlow operations
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        advantages = discounted_cumulative_sums(deltas, self.gamma * self.lam)
        self.advantage_buffer[path_slice].assign(advantages)

        # Calculate returns using TensorFlow operations
        returns = discounted_cumulative_sums(rewards, self.gamma)[:-1]
        self.return_env_buffer[path_slice].assign(returns)

        self.trajectory_start_index = self.pointer

    def get(self):
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = tf.reduce_mean(self.advantage_buffer), tf.math.reduce_std(self.advantage_buffer)
        self.advantage_buffer.assign((self.advantage_buffer - advantage_mean) / advantage_std)

        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_env_buffer,
            self.logprobability_buffer
        )


def mlp(x, sizes, activation=tf.keras.activations.tanh, output_activation=None):  # Build a feedforward neural network
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)

    return tf.keras.layers.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobabilities(action_logits_AI_agent, a):
    # Compute the log-probabilities of taking actions a by using
    # the action_logits_AI_agent (i.e. the output of the actor) (计算动作 a 的对数概率)
    logprobabilities_all = tf.nn.log_softmax(action_logits_AI_agent)
    logprobability = tf.reduce_sum(tf.one_hot(a, num_actions) * logprobabilities_all, axis=1)
    return logprobability


def _sample_action(_logits):
    return tf.cast(tf.squeeze(tf.random.categorical(_logits, 1, seed=seed_generator), axis=1), tf.int32)


@tf.function  # Sample action_AI_agent from actor
def sample_action(observation_AI):
    _logits = [actor(i) for i in observation_AI]
    _actions = [_sample_action(i) for i in _logits]
    return _logits, _actions


@tf.function  # Train the policy by maxizing the PPO-Clip objective
def train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):  # 训练策略网络
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(logprobabilities(actor(observation_buffer), action_buffer) - logprobability_buffer)
        min_advantage = tf.where(advantage_buffer > 0, (1 + clip_ratio) * advantage_buffer,
                                 (1 - clip_ratio) * advantage_buffer)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage))
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(logprobability_buffer - logprobabilities(actor(observation_buffer), action_buffer))
    kl = tf.reduce_sum(kl)
    return kl


@tf.function  # Train the value function by regression on mean-squared error
def train_value_function(observation_buffer, return_env_buffer):  # 训练价值函数
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_env_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    _env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, reward, _done, info = _env.step(data)
            remote.send((obs, reward, _done, info))
        elif cmd == "reset":
            obs = _env.reset()
            remote.send(obs)
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError


class EnvWrapper:
    def __init__(self, env_fn, _num_envs, max_steps_per_epoch):
        self._num_envs = _num_envs
        self.max_steps_per_epoch = max_steps_per_epoch
        self.current_step = np.zeros(_num_envs, dtype=int)
        self.pipes = []
        self.workers = []

        for _ in range(_num_envs):
            parent_pipe, child_pipe = Pipe()
            process = Process(target=worker, args=(child_pipe, parent_pipe, Wrapper(env_fn)))
            process.daemon = True
            process.start()
            child_pipe.close()

            self.pipes.append(parent_pipe)
            self.workers.append(process)

    def step(self, actions_hm_ai):

        for pipe, action in zip(self.pipes, actions_hm_ai):
            pipe.send(("step", action))

        results = [pipe.recv() for pipe in self.pipes]
        obs, rewards_sparse, rewards_shape, dones = zip(*results)

        for idx, _done in enumerate(dones):
            self.current_step[idx] += 1

            if _done or self.current_step[idx] >= self.max_steps_per_epoch:
                self.pipes[idx].send(("reset", None))
                obs[idx] = self.pipes[idx].recv()
                self.current_step[idx] = 0

        return obs, rewards_sparse, rewards_shape, dones

    def reset(self):
        for pipe in self.pipes:
            pipe.send(("reset", None))
        return [pipe.recv() for pipe in self.pipes]

    def close(self):
        for pipe in self.pipes:
            pipe.send(("close", None))
        for _worker in self.workers:
            _worker.join()


class Wrapper:
    def __init__(self, fn):
        self.fn = fn

    def x(self):
        return self.fn()


def caster(x):
    return tf.cast(tf.reshape(np.array(x), (1, -1)), dtype=tf.float32)


def get_observations(_obs_dict):
    _both, _other = [], []
    for i in _obs_dict:
        _both.append(i["both_agent_obs"])
        _other.append(i["other_agent_env_idx"])

    return _both, _other


def get_agent_obs(_both, _other):
    obs_ai, obs_hm = [], []
    for i, j in zip(_both, _other):
        obs_ai.append(caster(i[1 - j]))
        obs_hm.append(caster(i[j]))

    return obs_ai, obs_hm


def get_hm_action(obs_hm):
    def _get_hm_action(_obs_hm):
        return tf.argmax(tf.nn.softmax(bc_model_train(_obs_hm, training=False)), axis=1)

    return list(map(_get_hm_action, obs_hm))


def get_agent_value(agent):
    def _get_agent_value(_agent):
        return tf.keras.backend.get_value(_agent[0])

    return list(map(_get_agent_value, agent))


if __name__ == '__main__':
    from func_nn_ppo import func_nn_ppo
    from HyperParameters import *

    # ----
    seed_generator = tf.random.set_seed(1337)
    tf.config.run_functions_eagerly(False)
    env = EnvWrapper(make_env, num_envs, max_steps_per_epoch=steps_per_epoch)

    """
    ## Initializations  
    """
    temp_env = make_env()
    observation_dimensions = temp_env.observation_space.shape[0]
    num_actions = temp_env.action_space.n
    obs_dict = env.reset()  # 得到 reset 函数返回的字典值
    # -- 把字典里面的各种物理量提取出来 --
    both_agent_obs, other_agent_env_idx = get_observations(obs_dict)
    observation_AI, observation_HM = get_agent_obs(both_agent_obs, other_agent_env_idx)

    episode_return_sparse = [0 for _ in range(num_envs)]
    episode_return_shaped = [0 for _ in range(num_envs)]
    episode_return_env = [0 for _ in range(num_envs)]
    episode_length = [0 for _ in range(num_envs)]

    buffer = [Buffer(observation_dimensions, steps_per_epoch) for _ in range(num_envs)]
    actor, critic = func_nn_ppo(observation_dimensions, num_actions)

    # Initialize the policy and the value function optimizers
    if bc_model_path_train == "./bc_runs_ireca/reproduce_train/cramped_room":
        policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_policy_cr)
        value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value_cr)
    elif bc_model_path_train == "./bc_runs_ireca/reproduce_train/asymmetric_advantages":
        policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_policy_aa)
        value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value_aa)

    """
    ## Train
    """

    # Pre-compute and set static parameters outside loops
    avg_return_shaped, avg_return_sparse, avg_return_env = [], [], []

    # Training Loop
    for epoch in range(epochs):
        time = datetime.now()
        sum_return_sparse = [0 for _ in range(num_envs)]
        sum_return_shaped = [0 for _ in range(num_envs)]
        sum_return_env = [0 for _ in range(num_envs)]
        sum_length = [0 for _ in range(num_envs)]
        num_episodes = [0 for _ in range(num_envs)]

        for t in range(steps_per_epoch):
            # Sample actions for agents
            action_logits_AI_agent, action_AI_agent = sample_action(observation_AI)
            action_HM_agent = get_hm_action(observation_HM)

            # Step the environment
            obs_dict_new, reward_sparse, reward_shaped, done, _ = env.step(
                list(zip(get_agent_value(action_AI_agent), get_agent_value(action_HM_agent)))
            )

            # Extract and reshape observations once at each step
            observation_AI, observation_HM = get_agent_obs(obs_dict_new, other_agent_env_idx)

            # Compute and accumulate rewards
            for i in range(num_envs):
                reward_env = reward_sparse[i] + max(0, 1 - t * learning_rate_reward_shaping) * reward_shaped[i]
                episode_return_sparse[i] += reward_sparse[i]
                episode_return_shaped[i] += reward_shaped[i]
                episode_return_env[i] += [reward_env]

                buffer[i].store(
                    observation_AI[i],
                    action_AI_agent[i],
                    reward_env,
                    critic(observation_AI[i]),
                    logprobabilities(action_logits_AI_agent, action_AI_agent)
                )
                buffer[i].last_value_tensor = tf.constant([0], dtype=tf.float32) if done else critic(observation_AI[i])
                if done[i] or (t == steps_per_epoch - 1):
                    buffer[i].finish_trajectory()
                    sum_return_sparse[i] += episode_return_sparse[i]
                    sum_return_shaped[i] += episode_return_shaped[i]
                    sum_return_env[i] += episode_return_env[i]
                    sum_length[i] += episode_length[i]
                    num_episodes[i] += 1

                # Reset environment and episode stats
                obs_dict = env.reset()
                observation_AI, observation_HM = get_agent_obs(obs_dict, other_agent_env_idx)

                sum_return_sparse = [0 for _ in range(num_envs)]
                sum_return_shaped = [0 for _ in range(num_envs)]
                sum_return_env = [0 for _ in range(num_envs)]
                sum_length = [0 for _ in range(num_envs)]
                num_episodes = [0 for _ in range(num_envs)]

        training_time = datetime.now()
        # Training policy with mini-batches
        for _ in range(iterations_train_policy):
            for _buffer in buffer:
                for obs_batch, act_batch, adv_batch, ret_batch, logp_batch in tf_get_mini_batches_gpu(
                        *_buffer.get(), batch_size
                ):
                    kl = train_policy(obs_batch, act_batch, logp_batch, adv_batch)
                    if kl > 1.5 * target_kl:
                        break
                    train_value_function(obs_batch, ret_batch)

        print(f'TIME ELAPSED on TRAINING in EPOCH {epoch}: {str((datetime.now() - training_time).total_seconds())}')

        # Summarize results after each epoch
        avg_return_shaped.append(sum_return_shaped / num_episodes)
        avg_return_sparse.append(sum_return_sparse / num_episodes)
        avg_return_env.append(sum_return_env / num_episodes)

        print(f'TIME ELAPSED on EPOC {epoch}: {str((datetime.now() - time).total_seconds())}')
