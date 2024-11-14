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
    # Convert buffers to GPU tensors
    obs_buf = tf.identity(tf.convert_to_tensor(obs_buf), device='/gpu:0')
    act_buf = tf.identity(tf.convert_to_tensor(act_buf), device='/gpu:0')
    adv_buf = tf.identity(tf.convert_to_tensor(adv_buf), device='/gpu:0')
    ret_buf = tf.identity(tf.convert_to_tensor(ret_buf), device='/gpu:0')
    logp_buf = tf.identity(tf.convert_to_tensor(logp_buf), device='/gpu:0')

    # Use Dataset API to batch and prefetch
    dataset = tf.data.Dataset.from_tensor_slices((
        obs_buf,
        act_buf,
        adv_buf,
        ret_buf,
        logp_buf,
    ))
    dataset = dataset.shuffle(buffer_size=len(obs_buf))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates (计算折扣累计和，用于计算奖励到期和优势估计)
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:  # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        self.size = size
        self.observation_buffer = tf.TensorArray(dtype=tf.float64, size=size)
        self.action_buffer = tf.TensorArray(dtype=tf.int64, size=size)
        self.advantage_buffer = tf.TensorArray(dtype=tf.float64, size=size)
        self.reward_env_buffer = tf.TensorArray(dtype=tf.float64, size=size)
        self.return_env_buffer = tf.TensorArray(dtype=tf.float64, size=size)
        self.value_buffer = tf.TensorArray(dtype=tf.float64, size=size)
        self.logprobability_buffer = tf.TensorArray(dtype=tf.float64, size=size)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation_AI, action_AI_agent, reward_env, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer = self.observation_buffer.write(self.pointer, observation_AI)
        self.action_buffer = self.action_buffer.write(self.pointer, action_AI_agent)
        self.reward_env_buffer = self.reward_env_buffer.write(self.pointer, reward_env)
        self.value_buffer = self.value_buffer.write(self.pointer, value)
        self.logprobability_buffer = self.logprobability_buffer.write(self.pointer, logprobability)
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Compute advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = tf.concat([self.reward_env_buffer.stack()[path_slice], [last_value]], 0)
        values = tf.concat([self.value_buffer.stack()[path_slice], [last_value]], 0)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer = self.advantage_buffer.scatter(tf.range(path_slice.start, path_slice.stop),
                                                              discounted_cumulative_sums(deltas, self.gamma * self.lam))
        self.return_env_buffer = self.return_env_buffer.scatter(tf.range(path_slice.start, path_slice.stop),
                                                                discounted_cumulative_sums(rewards, self.gamma)[:-1])
        self.trajectory_start_index = self.pointer

    def get(self):  # Retrieve all data and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean = tf.reduce_mean(self.advantage_buffer.stack())
        advantage_std = tf.math.reduce_std(self.advantage_buffer.stack())
        normalized_advantages = (self.advantage_buffer.stack() - advantage_mean) / advantage_std
        return (
            self.observation_buffer.stack(),
            self.action_buffer.stack(),
            normalized_advantages,
            self.return_env_buffer.stack(),
            self.logprobability_buffer.stack(),
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

both_agent_obs_tensor = tf.TensorArray(tf.float32, size=2)  # Assuming both_agent_obs has 2 elements
both_agent_obs_tensor = both_agent_obs_tensor.unstack(tf.convert_to_tensor(both_agent_obs, dtype=tf.dtypes.float32))

observation_AI = both_agent_obs_tensor.read(1 - other_agent_env_idx)
observation_HM = both_agent_obs_tensor.read(other_agent_env_idx)

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

avg_return_shaped = []
avg_return_sparse = []
avg_return_env = []

for epoch in range(epochs):

    sum_return_sparse = 0
    sum_return_shaped = 0
    sum_return_env = 0
    sum_length = 0
    num_episodes = 0

    observation_AI = tf.reshape(observation_AI, (1, -1))
    observation_HM = tf.reshape(observation_HM, (1, -1))

    for t in range(steps_per_epoch):

        count_step += 1

        action_logits_AI_agent, action_AI_agent = sample_action(observation_AI)
        action_logits_bc_agent = bc_model_train(observation_HM, training=False)

        action_probs_bc_agent = tf.nn.softmax(action_logits_bc_agent)
        action_HM_agent = tf.argmax(action_probs_bc_agent, axis=1)  # 不用随机策略

        # - Convert TensorFlow tensors to Python integers
        action_AI_agent_np = action_AI_agent.numpy().tolist()[0]
        action_HM_agent_np = action_HM_agent.numpy().tolist()[0]
        action_np = [action_AI_agent_np, action_HM_agent_np]

        # observation_AI_new, reward_env, done, _, _ = env.step(action_AI_agent[0].numpy())
        obs_dict_new, reward_sparse, reward_shaped, done, _ = env.step(action_np)

        observation_AI_new = tf.reshape(obs_dict_new["both_agent_obs"][1 - other_agent_env_idx], (1, -1))
        observation_HM_new = tf.reshape(obs_dict_new["both_agent_obs"][other_agent_env_idx], (1, -1))

        coeff_reward_shaped = max(0, 1 - count_step * learning_rate_reward_shaping)
        reward_env = reward_sparse + coeff_reward_shaped * reward_shaped

        episode_return_sparse += reward_sparse
        episode_return_shaped += reward_shaped
        episode_return_env += reward_env
        episode_length += 1

        # Get the value and log-probability of the action_AI_agent (获取动作的价值和对数概率)
        value_t = critic(observation_AI)
        logprobability_t = logprobabilities(action_logits_AI_agent, action_AI_agent)

        # Store obs, act, rew, v_t, logp_pi_t (存储观测、动作、奖励、价值和对数概率)
        buffer.store(observation_AI, action_AI_agent, reward_env, value_t, logprobability_t)

        # Update the observation_AI (更新观测)
        observation_AI = observation_AI_new
        observation_HM = observation_HM_new

        # Finish trajectory if reached to a terminal state (如果达到终止状态则结束轨迹)
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(observation_AI.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            sum_return_shaped += episode_return_shaped
            sum_return_sparse += episode_return_sparse
            sum_return_env += episode_return_env
            sum_length += episode_length
            num_episodes += 1
            obs_dict = env.reset()
            observation_AI = tf.reshape(obs_dict["both_agent_obs"][1 - other_agent_env_idx], (1, -1))
            observation_HM = tf.reshape(obs_dict["both_agent_obs"][other_agent_env_idx], (1, -1))
            episode_return_shaped, episode_return_sparse, episode_return_env, episode_length = 0, 0, 0, 0

    (obs_buf, act_buf, adv_buf, ret_buf, logp_buf) = buffer.get()

    for _ in range(iterations_train_policy):
        for (
                obs_batch, act_batch, adv_batch, ret_batch, logp_batch
        ) in tf_get_mini_batches(
            obs_buf, act_buf, adv_buf,  ret_buf, logp_buf, batch_size
        ):
            kl = train_policy(obs_batch, act_batch, logp_batch, adv_batch)
            if kl > 1.5 * target_kl:
                break
            train_value_function(obs_batch, ret_batch)

    print(f" [ppobc] Epoch: {epoch}. Mean Length: {sum_length / num_episodes}")
    print(
        f" Mean sparse: {sum_return_sparse / num_episodes}. "
        f"Mean shaped: {sum_return_shaped / num_episodes}. "
        f"Mean Env: {sum_return_env / num_episodes}. "
    )

    avg_return_shaped.append(sum_return_shaped / num_episodes)
    avg_return_sparse.append(sum_return_sparse / num_episodes)
    avg_return_env.append(sum_return_env / num_episodes)


if bc_model_path_train == "./bc_runs_ccima/reproduce_train/cramped_room":
    actor.save_weights("model_cr_actor_ppobc.h5")
    critic.save_weights("model_cr_critic_ppobc.h5")
elif bc_model_path_train == "./bc_runs_ccima/reproduce_train/asymmetric_advantages":
    actor.save_weights("model_aa_actor_ppobc.h5")
    critic.save_weights("model_aa_critic_ppobc.h5")
