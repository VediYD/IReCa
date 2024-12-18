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


def tf_get_mini_batches(_obs_buf, _act_buf, _adv_buf, _ret_buf, _logp_buf, _batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        _obs_buf,
        _act_buf,
        _adv_buf,
        _ret_buf,
        _logp_buf,
    ))

    dataset = dataset.shuffle(buffer_size=len(_obs_buf))  # 打乱数据
    dataset = dataset.batch(_batch_size)  # 按批次分割数据
    return dataset


def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:  # Buffer for storing trajectories (存储轨迹的缓冲区)
    def __init__(self, _observation_dimensions, size, _gamma=0.99, _lam=0.95):  # Buffer initialization (缓冲区初始化)
        self.observation_buffer = np.zeros((size, _observation_dimensions), dtype=np.float32)
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_env_buffer = np.zeros(size, dtype=np.float32)
        self.return_env_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.log_probability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = _gamma, _lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, _observation_ai, _action_ai_agent, _reward_env, value, log_probability):
        self.observation_buffer[self.pointer] = _observation_ai
        self.action_buffer[self.pointer] = _action_ai_agent
        self.reward_env_buffer[self.pointer] = _reward_env
        self.value_buffer[self.pointer] = value
        self.log_probability_buffer[self.pointer] = log_probability
        self.pointer += 1

    def finish_trajectory(self, _last_value=0):
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_env_buffer[path_slice], _last_value)
        values = np.append(self.value_buffer[path_slice], _last_value)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]  # 计算优势估计
        self.advantage_buffer[path_slice] = discounted_cumulative_sums(deltas, self.gamma * self.lam)
        self.return_env_buffer[path_slice] = discounted_cumulative_sums(rewards, self.gamma)[:-1]  # 计算奖励到期
        self.trajectory_start_index = self.pointer

    def get(self):
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_env_buffer,
            self.log_probability_buffer,
        )


def mlp(x, sizes, output_activation=None):  # Build a feedforward neural network
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    return tf.keras.layers.Dense(units=sizes[-1], activation=output_activation)(x)


def log_probabilities(_action_logits_ai_agent, a):
    log_probabilities_all = tf.nn.log_softmax(_action_logits_ai_agent)
    log_probability = tf.reduce_sum(tf.one_hot(a, num_actions) * log_probabilities_all, axis=1)
    return log_probability


@tf.function  # Sample action_AI_agent from actor
def sample_action(_observation_ai):
    _action_logits_ai_agent = actor(_observation_ai)
    _action_ai_agent = tf.squeeze(tf.random.categorical(_action_logits_ai_agent, 1, seed=seed_generator), axis=1)
    return _action_logits_ai_agent, _action_ai_agent


@tf.function  # Train the policy by maxizing the PPO-Clip objective
def train_policy(observation_buffer, action_buffer, log_probability_buffer, advantage_buffer):  # 训练策略网络
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(log_probabilities(actor(observation_buffer), action_buffer) - log_probability_buffer)
        min_advantage = tf.where(advantage_buffer > 0, (1 + clip_ratio) * advantage_buffer,
                                 (1 - clip_ratio) * advantage_buffer)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage))
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    _kl = tf.reduce_mean(log_probability_buffer - log_probabilities(actor(observation_buffer), action_buffer))
    _kl = tf.reduce_sum(_kl)
    return _kl


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

obs_dict = env.reset()

both_agent_obs = obs_dict["both_agent_obs"]
other_agent_env_idx = obs_dict["other_agent_env_idx"]

observation_AI = np.array(both_agent_obs[1 - other_agent_env_idx])
observation_HM = np.array(both_agent_obs[other_agent_env_idx])

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

        #         action_logits_bc_agent = bc_model.predict(observation_HM, verbose=0)
        #         action_logits_bc_agent = bc_model.predict_on_batch(observation_HM)
        action_logits_bc_agent = bc_model_train(observation_HM, training=False)

        action_probs_bc_agent = tf.nn.softmax(action_logits_bc_agent)
        action_HM_agent = tf.argmax(action_probs_bc_agent, axis=1)  # 不用随机策略

        # - Convert TensorFlow tensors to Python integers
        action_AI_agent_np = action_AI_agent.numpy().tolist()[0]
        action_HM_agent_np = action_HM_agent.numpy().tolist()[0]
        action_np = [action_AI_agent_np, action_HM_agent_np]

        # observation_AI_new, reward_env, done, _, _ = env.step(action_AI_agent[0].numpy())
        print(action_np)
        obs_dict_new, reward_sparse, reward_shaped, done, _ = env.step(action_np)
        print(reward_sparse, reward_shaped)

        observation_AI_new = tf.reshape(obs_dict_new["both_agent_obs"][1 - other_agent_env_idx], (1, -1))
        observation_HM_new = tf.reshape(obs_dict_new["both_agent_obs"][other_agent_env_idx], (1, -1))

        coeff_reward_shaped = max(0, 1 - count_step * learning_rate_reward_shaping)
        reward_env = reward_sparse + coeff_reward_shaped * reward_shaped
        #         reward_env = tf.add(reward_sparse, tf.multiply(coeff_reward_shaped, reward_shaped))

        episode_return_sparse += reward_sparse
        episode_return_shaped += reward_shaped
        episode_return_env += reward_env
        episode_length += 1

        # Get the value and log-probability of the action_AI_agent (获取动作的价值和对数概率)
        value_t = critic(observation_AI)
        log_probability_t = log_probabilities(action_logits_AI_agent, action_AI_agent)

        # Store obs, act, rew, v_t, logp_pi_t (存储观测、动作、奖励、价值和对数概率)
        buffer.store(observation_AI, action_AI_agent, reward_env, value_t, log_probability_t)

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
        for obs_batch, act_batch, adv_batch, ret_batch, logp_batch in tf_get_mini_batches(obs_buf, act_buf, adv_buf,
                                                                                          ret_buf, logp_buf,
                                                                                          batch_size):

            kl = train_policy(obs_batch, act_batch, logp_batch, adv_batch)
            if kl > 1.5 * target_kl:
                break
            train_value_function(obs_batch, ret_batch)

    print(f" [ppobc] Epoch: {epoch}. Mean Length: {sum_length / num_episodes}")
    print(
        f" Mean sparse: {sum_return_sparse / num_episodes}. Mean shaped: {sum_return_shaped / num_episodes}. Mean Env: {sum_return_env / num_episodes}. ")

    avg_return_shaped.append(sum_return_shaped / num_episodes)
    avg_return_sparse.append(sum_return_sparse / num_episodes)
    avg_return_env.append(sum_return_env / num_episodes)

#     if ((epoch+1) % 20 == 0) or ((epoch+1) == epochs) :
#         np.save('./data_tmp/data_ppobc_return_shaped.npy', avg_return_shaped)
#         np.save('./data_tmp/data_ppobc_return_sparse.npy', avg_return_sparse)
#         np.save('./data_tmp/data_ppobc_return_env.npy',    avg_return_env)


if bc_model_path_train == "./bc_runs_ccima/reproduce_train/cramped_room":
    actor.save_weights("model_cr_actor_ppobc.h5")
    critic.save_weights("model_cr_critic_ppobc.h5")
elif bc_model_path_train == "./bc_runs_ccima/reproduce_train/asymmetric_advantages":
    actor.save_weights("model_aa_actor_ppobc.h5")
    critic.save_weights("model_aa_critic_ppobc.h5")

# # ----Plot Figure----
# plt.figure()
# plt.plot(avg_return_env, markerfacecolor='none')
# # plt.xlabel('Index of data samples')
# plt.ylabel('avg_return_env')
# plt.legend(fontsize=12, loc='lower right')
# plt.savefig('./figs/ppobc_avg_return_env.pdf', format='pdf')  


# plt.show()
