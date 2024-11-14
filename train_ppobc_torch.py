import os

# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用哪块GPU

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from func_nn_ppo_torch import ActorCritic
from HyperParameters import *

# ----
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
else:
    torch.manual_seed(1337)

"""
## Functions and class
"""


def get_mini_batches(obs_buf, act_buf, adv_buf, ret_buf, logp_buf, batch_size):
    # convert the buffers to tensors
    obs_tensor = torch.tensor(obs_buf, dtype=torch.float32)
    act_tensor = torch.tensor(act_buf, dtype=torch.float32)
    adv_tensor = torch.tensor(adv_buf, dtype=torch.float32)
    ret_tensor = torch.tensor(ret_buf, dtype=torch.float32)
    logp_tensor = torch.tensor(logp_buf, dtype=torch.float32)

    # create a dataset and data-loader object
    dataset = TensorDataset(obs_tensor, act_tensor, adv_tensor, ret_tensor, logp_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing
    # rewards-to-go and advantage estimates (计算折扣累计和，用于计算奖励到期和优势估计)
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95, device='gpu'):
        self.device = device  # Store the device ('cpu' or 'cuda')
        self.observation_buffer = torch.zeros((size, observation_dimensions), dtype=torch.float32, device=self.device)
        self.action_buffer = torch.zeros(size, dtype=torch.int32, device=self.device)
        self.advantage_buffer = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.reward_env_buffer = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.return_env_buffer = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.value_buffer = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.logprobability_buffer = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation_AI, action_AI_agent, reward_env, value, logprobability):
        # Convert inputs to torch tensors on the specified device
        self.observation_buffer[self.pointer] = torch.tensor(observation_AI, dtype=torch.float32, device=self.device)
        self.action_buffer[self.pointer] = torch.tensor(action_AI_agent, dtype=torch.int32, device=self.device)
        self.reward_env_buffer[self.pointer] = torch.tensor(reward_env, dtype=torch.float32, device=self.device)
        self.value_buffer[self.pointer] = torch.tensor(value, dtype=torch.float32, device=self.device)
        self.logprobability_buffer[self.pointer] = torch.tensor(logprobability, dtype=torch.float32, device=self.device)
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = torch.cat(
            (self.reward_env_buffer[path_slice], torch.tensor([last_value], dtype=torch.float32, device=self.device)))
        values = torch.cat(
            (self.value_buffer[path_slice], torch.tensor([last_value], dtype=torch.float32, device=self.device)))

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(deltas, self.gamma * self.lam)

        self.return_env_buffer[path_slice] = self.discounted_cumulative_sums(rewards, self.gamma)[:-1]
        self.trajectory_start_index = self.pointer

    def get(self):
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean = torch.mean(self.advantage_buffer)
        advantage_std = torch.std(self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_env_buffer,
            self.logprobability_buffer,
        )

    @staticmethod
    def discounted_cumulative_sums(x, discount):
        # Calculate discounted cumulative sums (similar to tf's implementation)
        result = torch.zeros_like(x)
        result[-1] = x[-1]
        for i in reversed(range(len(x) - 1)):
            result[i] = x[i] + discount * result[i + 1]
        return result


class MLP(nn.Module):
    def __init__(self, sizes, activation=nn.Tanh, output_activation=None):
        super(MLP, self).__init__()

        # Define layers
        self.layers = nn.ModuleList()
        input_size = sizes[0]

        for size in sizes[1:-1]:
            self.layers.append(nn.Linear(input_size, size))
            self.layers.append(activation())
            input_size = size

        # Final output layer
        self.layers.append(nn.Linear(input_size, sizes[-1]))
        if output_activation:
            self.layers.append(output_activation())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def logprobabilities(action_logits_AI_agent, a, num_actions):
    logprobabilities_all = F.log_softmax(action_logits_AI_agent, dim=1)
    a = a.long()
    logprobability = logprobabilities_all.gather(1, a.unsqueeze(1)).squeeze(1)
    return logprobability


def sample_action(observation_AI, actor, seed_generator=None):
    # get logits from the actor model
    action_logits_AI_agent = actor(observation_AI)

    # create a Categorical distribution and sample an action
    distribution = torch.distributions.Categorical(logits=action_logits_AI_agent)
    action_AI_agent = distribution.sample()

    return action_logits_AI_agent, action_AI_agent


def train_policy(
        observation_buffer,
        action_buffer,
        logprobability_buffer,
        advantage_buffer,
        actor,
        policy_optimizer,
        clip_ratio
):
    observation_buffer = observation_buffer.to(actor.device)
    action_buffer = action_buffer.to(actor.device)
    logprobability_buffer = logprobability_buffer.to(actor.device)
    advantage_buffer = advantage_buffer.to(actor.device)

    actor.train()
    # forward pass
    action_logits_AI_agent = actor(observation_buffer)
    logprobabilities_all = F.log_softmax(action_logits_AI_agent, dim=1)
    ratio = torch.exp(logprobabilities_all.gather(1, action_buffer.unsqueeze(1)) - logprobability_buffer.unsqueeze(1))

    # calculate the advantage
    min_advantage = torch.where(
        advantage_buffer > 0, (1 + clip_ratio) * advantage_buffer, (1 - clip_ratio) * advantage_buffer
    )

    # PPO-Clip objective
    policy_loss = -torch.mean(torch.minimum(ratio * advantage_buffer, min_advantage))

    # compute the gradients
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # compute kl divergence
    kl = torch.mean(logprobability_buffer - logprobabilities_all.gather(1, action_buffer.unsqueeze(1)))
    kl = torch.sum(kl)

    return kl


def train_value_function(observation_buffer, return_env_buffer, critic, value_optimizer):
    observation_buffer = observation_buffer.to(critic.device)
    return_env_buffer = return_env_buffer.to(critic.device)

    critic.train()

    # forward pass
    predicted_value = critic(observation_buffer)
    value_loss = F.mse_loss(predicted_value, return_env_buffer)

    # zero gradients, backpropagate, and update the critic
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    return value_loss


"""
## Initializations  
"""
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n
# print('\n')
# print('observation_dimensions:', observation_dimensions)
# print('num_actions:', num_actions)
# # print('\nenv.action_space:', env.action_space)
# print('\n')

obs_dict = env.reset()  # 得到 reset 函数返回的字典值
# -- 把字典里面的各种物理量提取出来 --
both_agent_obs = obs_dict["both_agent_obs"]  # 获取两个智能体的观察值
# agent_obs_0 = both_agent_obs[0]  # 第一个智能体的观察值
# agent_obs_1 = both_agent_aobs[1]  # 第二个智能体的观察值
# overcooked_state = obs_dict["overcooked_state"] # 获取当前的Overcooked状态
other_agent_env_idx = obs_dict["other_agent_env_idx"]  # 获取另一个智能体的环境索引
# print('\n')
# # print('=====> both_agent_obs is:', both_agent_obs)
# # print('=====> agent_obs_0 is:', agent_obs_0)
# # print('=====> agent_obs_1 is:', agent_obs_1)
# # print('=====> overcooked_state is:', overcooked_state)
# print('=====> other_agent_env_idx is:', other_agent_env_idx)
# print('\n')

# ACTION_TO_CHAR = {
#     Direction.NORTH: "↑",
#     Direction.SOUTH: "↓",
#     Direction.EAST: "→",
#     Direction.WEST: "←",
#     STAY: "stay",
#     INTERACT: INTERACT,
# }


# -- 现在你可以使用这些变量进行后续的操作，比如决定哪个智能体行动，或者基于当前状态进行策略计算等
observation_AI = np.array(both_agent_obs[1 - other_agent_env_idx])
observation_HM = np.array(both_agent_obs[other_agent_env_idx])
episode_return_sparse, episode_return_shaped = 0, 0
episode_return_env, episode_length = 0, 0
count_step = 0

# ---- (CartPole) ----
# env = gym.make("CartPole-v1")
# observation_dimensions = env.observation_space.shape[0]
# num_actions = env.action_space.n
# # Initialize
# observation_AI, _ = env.reset()
# # print('=====> observation_AI init is:', observation_AI)
# episode_return_env, episode_length = 0, 0
# -------- end of env configuration --------

buffer = Buffer(observation_dimensions, steps_per_epoch)  # Initialize the buffer (# 初始化缓冲区)
model = ActorCritic(observation_dimensions, num_actions)
actor, critic =

# observation_input = tf.keras.Input(shape=(observation_dimensions,), dtype="float32")
# action_logits_AI_agent = mlp(observation_input, list(mlp_hidden_sizes) + [num_actions])
# actor = tf.keras.Model(inputs=observation_input, outputs=action_logits_AI_agent, name='actor_keras')
# value = tf.squeeze(mlp(observation_input, list(mlp_hidden_sizes) + [1]), axis=1)
# critic = tf.keras.Model(inputs=observation_input, outputs=value, name='critic_keras')

# actor.summary()
# critic.summary()

# Initialize the policy and the value function optimizers
if bc_model_path_train == "./bc_runs_ireca/reproduce_train/cramped_room":
    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate_policy_cr)
    value_optimizer = optim.Adam(value.parameters(), lr=learning_rate_value_cr)
elif bc_model_path_train == "./bc_runs_ireca/reproduce_train/asymmetric_advantages":
    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate_policy_aa)
    value_optimizer = optim.Adam(value.parameters(), lr=learning_rate_value_aa)

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

    observation_AI = observation_AI.reshape(1, -1)
    observation_HM = observation_HM.reshape(1, -1)

    for t in range(steps_per_epoch):
        count_step += 1

        action_logits_AI_agent, action_AI_agent = sample_action(observation_AI)

        action_logits_bc_agent = bc_model_train(observation_HM, training=False)

        action_probs_bc_agent = torch.softmax(action_logits_bc_agent, dim=1)
        action_HM_agent = torch.argmax(action_probs_bc_agent, dim=1)

        action_AI_agent_np = action_AI_agent.item()  # .item() is used to get the scalar from a tensor
        action_HM_agent_np = action_HM_agent.item()
        action_np = [action_AI_agent_np, action_HM_agent_np]

        obs_dict_new, reward_sparse, reward_shaped, done, _ = env.step(action_np)

        observation_AI_new = torch.tensor(
            obs_dict_new["both_agent_obs"][1 - other_agent_env_idx], dtype=torch.float32
        ).reshape(1, -1)

        observation_HM_new = torch.tensor(
            obs_dict_new["both_agent_obs"][other_agent_env_idx], dtype=torch.float32
        ).reshape(1, -1)

        coeff_reward_shaped = max(0, 1 - count_step * learning_rate_reward_shaping)
        reward_env = reward_sparse + coeff_reward_shaped * reward_shaped

        episode_return_sparse += reward_sparse
        episode_return_shaped += reward_shaped
        episode_return_env += reward_env
        episode_length += 1

        value_t = critic(observation_AI)
        logprobability_t = logprobabilities(action_logits_AI_agent, action_AI_agent)

        buffer.store(observation_AI, action_AI_agent, reward_env, value_t, logprobability_t)

        observation_AI = observation_AI_new
        observation_HM = observation_HM_new

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
            observation_AI = torch.tensor(
                obs_dict["both_agent_obs"][1 - other_agent_env_idx], dtype=torch.float32
            ).reshape(1, -1)
            observation_HM = torch.tensor(
                obs_dict["both_agent_obs"][other_agent_env_idx], dtype=torch.float32
            ).reshape(1, -1)
            episode_return_shaped, episode_return_sparse, episode_return_env, episode_length = 0, 0, 0, 0

    obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buffer.get()

    for _ in range(iterations_train_policy):
        for obs_batch, act_batch, adv_batch, ret_batch, logp_batch in get_mini_batches(
                obs_buf, act_buf, adv_buf, ret_buf, logp_buf, batch_size
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
