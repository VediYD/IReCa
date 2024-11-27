import os

import numpy as np
import tensorflow as tf
import scipy.signal
import matplotlib.pyplot as plt

from func_nn_ppo import func_nn_ppo
from HyperParameters import *

# ----

seed_generator = tf.random.set_seed(1337)

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
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        self.observation_buffer = tf.Variable(tf.zeros((size, observation_dimensions), dtype=tf.float32))
        self.action_buffer = tf.Variable(tf.zeros(size, dtype=tf.int32))
        self.advantage_buffer = tf.Variable(tf.zeros(size, dtype=tf.float32))
        self.reward_env_buffer = tf.Variable(tf.zeros(size, dtype=tf.float32))
        self.return_env_buffer = tf.Variable(tf.zeros(size, dtype=tf.float32))
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

    def finish_trajectory(self, last_value=0):
        path_slice = slice(self.trajectory_start_index, self.pointer)

        # Slice relevant parts of the buffers
        rewards = self.reward_env_buffer[path_slice]
        values = self.value_buffer[path_slice]

        # Append `last_value` using tf.concat
        rewards = tf.concat([rewards, tf.convert_to_tensor([last_value], dtype=rewards.dtype)], axis=0)
        values = tf.concat([values, tf.convert_to_tensor([last_value], dtype=values.dtype)], axis=0)

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


@tf.function  # Sample action_AI_agent from actor
def sample_action(observation_AI):
    action_logits_AI_agent = actor(observation_AI)
    action_AI_agent = tf.squeeze(tf.random.categorical(action_logits_AI_agent, 1, seed=seed_generator), axis=1)
    action_AI_agent = tf.cast(action_AI_agent, tf.int32)
    return action_logits_AI_agent, action_AI_agent


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


"""
## Initializations  
"""
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n
obs_dict = env.reset()  # 得到 reset 函数返回的字典值
# -- 把字典里面的各种物理量提取出来 --
both_agent_obs = obs_dict["both_agent_obs"]  # 获取两个智能体的观察值
other_agent_env_idx = obs_dict["other_agent_env_idx"]  # 获取另一个智能体的环境索引

observation_AI = np.array(both_agent_obs[1 - other_agent_env_idx])
observation_HM = np.array(both_agent_obs[other_agent_env_idx])

observation_AI = tf.cast(tf.reshape(observation_AI, (1, -1)), dtype=tf.float32)
observation_HM = tf.cast(tf.reshape(observation_HM, (1, -1)), dtype=tf.float32)

episode_return_sparse, episode_return_shaped = 0, 0
episode_return_env, episode_length = 0, 0
count_step = 0

buffer = Buffer(observation_dimensions, steps_per_epoch)  # Initialize the buffer (# 初始化缓冲区)
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
observation_AI = tf.reshape(observation_AI, (1, -1))
observation_HM = tf.reshape(observation_HM, (1, -1))

from datetime import datetime


# Training Loop
for epoch in range(epochs):
    t = datetime.now()
    sum_return_sparse, sum_return_shaped, sum_return_env, sum_length, num_episodes = 0, 0, 0, 0, 0

    for t in range(steps_per_epoch):
        # Sample actions for agents
        action_logits_AI_agent, action_AI_agent = sample_action(observation_AI)
        action_HM_agent = tf.argmax(tf.nn.softmax(bc_model_train(observation_HM, training=False)), axis=1)

        # Step the environment
        obs_dict_new, reward_sparse, reward_shaped, done, _ = env.step(
            [tf.keras.backend.get_value(action_AI_agent[0]), tf.keras.backend.get_value(action_HM_agent[0])]
        )

        # Extract and reshape observations once at each step
        observation_AI, observation_HM = (
            tf.reshape(obs_dict_new["both_agent_obs"][1 - other_agent_env_idx], (1, -1)),
            tf.reshape(obs_dict_new["both_agent_obs"][other_agent_env_idx], (1, -1))
        )

        # Compute and accumulate rewards
        reward_env = reward_sparse + max(0, 1 - count_step * learning_rate_reward_shaping) * reward_shaped
        episode_return_sparse, episode_return_shaped, episode_return_env = (
            episode_return_sparse + reward_sparse,
            episode_return_shaped + reward_shaped,
            episode_return_env + reward_env,
        )

        # Store necessary data in buffer
        buffer.store(observation_AI, action_AI_agent, reward_env, critic(observation_AI),
                     logprobabilities(action_logits_AI_agent, action_AI_agent))

        # Handle terminal state and reset
        if done or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(observation_AI)
            buffer.finish_trajectory(last_value)
            sum_return_sparse += episode_return_sparse
            sum_return_shaped += episode_return_shaped
            sum_return_env += episode_return_env
            sum_length += episode_length
            num_episodes += 1

            # Reset environment and episode stats
            obs_dict = env.reset()
            observation_AI, observation_HM = (
                tf.reshape(obs_dict["both_agent_obs"][1 - other_agent_env_idx], (1, -1)),
                tf.reshape(obs_dict["both_agent_obs"][other_agent_env_idx], (1, -1))
            )
            episode_return_sparse, episode_return_shaped, episode_return_env, episode_length = 0, 0, 0, 0

    # Training policy with mini-batches
    for _ in range(iterations_train_policy):
        for obs_batch, act_batch, adv_batch, ret_batch, logp_batch in tf_get_mini_batches_gpu(
            *buffer.get(), batch_size
        ):
            kl = train_policy(obs_batch, act_batch, logp_batch, adv_batch)
            if kl > 1.5 * target_kl:
                break
            train_value_function(obs_batch, ret_batch)

    # Summarize results after each epoch
    avg_return_shaped.append(sum_return_shaped / num_episodes)
    avg_return_sparse.append(sum_return_sparse / num_episodes)
    avg_return_env.append(sum_return_env / num_episodes)

    print('TIME ELAPSED on EPOC: ', (datetime.now() - t).microseconds)
