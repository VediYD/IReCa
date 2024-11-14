import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, observation_dimensions, num_actions, l1_reg=1e-4, l2_reg=2e-4):
        super(ActorCritic, self).__init__()

        # Regularization through weight decay (L2)
        self.l2_reg = l2_reg

        # Actor Network
        self.actor_conv1 = nn.Conv2d(1, 25, kernel_size=(5, 5), stride=1, padding='same')
        self.actor_conv2 = nn.Conv2d(25, 25, kernel_size=(3, 3), stride=1, padding='same')
        self.actor_conv3 = nn.Conv2d(25, 25, kernel_size=(3, 3), stride=1, padding='same')
        self.actor_fc1 = nn.Linear(12 * 8 * 25, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_out = nn.Linear(64, num_actions)

        # Critic Network
        self.critic_conv1 = nn.Conv2d(1, 25, kernel_size=(5, 5), stride=1, padding='same')
        self.critic_conv2 = nn.Conv2d(25, 25, kernel_size=(3, 3), stride=1, padding='same')
        self.critic_conv3 = nn.Conv2d(25, 25, kernel_size=(3, 3), stride=1, padding='same')
        self.critic_fc1 = nn.Linear(12 * 8 * 25, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_out = nn.Linear(64, 1)

    def forward(self, x):
        # Actor forward pass
        x_actor = x.view(-1, 1, 12, 8)  # Reshape input for Conv2D
        x_actor = F.tanh(self.actor_conv1(x_actor))
        x_actor = F.tanh(self.actor_conv2(x_actor))
        x_actor = F.tanh(self.actor_conv3(x_actor))
        x_actor = x_actor.view(x_actor.size(0), -1)  # Flatten for dense layers
        x_actor = F.tanh(self.actor_fc1(x_actor))
        x_actor = F.tanh(self.actor_fc2(x_actor))
        action_logits = self.actor_out(x_actor)

        # Critic forward pass
        x_critic = x.view(-1, 1, 12, 8)  # Reshape input for Conv2D
        x_critic = F.tanh(self.critic_conv1(x_critic))
        x_critic = F.tanh(self.critic_conv2(x_critic))
        x_critic = F.tanh(self.critic_conv3(x_critic))
        x_critic = x_critic.view(x_critic.size(0), -1)  # Flatten for dense layers
        x_critic = F.tanh(self.critic_fc1(x_critic))
        x_critic = F.tanh(self.critic_fc2(x_critic))
        value = self.critic_out(x_critic)

        return action_logits, value



# observation_dimensions = 96 (12x8)
# num_actions = 5 (number of possible actions)
model = ActorCritic(observation_dimensions=96, num_actions=5)

# input (batch of 1)
x = torch.randn(1, 96)  # Flattened 12x8 input

# pass
action_logits, value = model(x)
print(action_logits, value)
