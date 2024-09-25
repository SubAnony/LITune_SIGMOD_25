import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.set_device(0)

class Context(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, device):
        super(Context, self).__init__()
        self.recurrent = nn.GRU(input_dim, hidden_sizes[0], batch_first=True)
        self.fc = nn.Linear(hidden_sizes[0], output_dim)
        self.device = device

    def forward(self, previous_action, previous_reward, pre_x):
        bsize = previous_action.size(0)
        hidden = self.init_recurrent(bsize)
        combined_input = torch.cat([previous_action, previous_reward, pre_x], dim=-1)
        _, hidden = self.recurrent(combined_input, hidden)
        context_vector = self.fc(hidden.squeeze(0))
        return context_vector

    def init_recurrent(self, bsize):
        return torch.zeros(1, bsize, self.recurrent.hidden_size).to(self.device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, context_module):
        super(Actor, self).__init__()
        self.context_module = context_module
        self.l1 = nn.Linear(state_dim + context_module.fc.out_features, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state, pre_infos):
        context_vector = self.context_module(*pre_infos)
        x = torch.cat([state, context_vector], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, context_module):
        super(Critic, self).__init__()
        self.context_module = context_module
        self.l1 = nn.Linear(state_dim + action_dim + context_module.fc.out_features, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action, pre_infos):
        context_vector = self.context_module(*pre_infos)
        sa = torch.cat([state, action, context_vector], 1)
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        return self.l3(x)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 40)
        self.l2 = nn.Linear(40, 30)
        self.l3 = nn.Linear(30, action_dim)

        self.max_action = max_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) 
        return self.max_action * torch.sigmoid(a)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 40)
        self.l2 = nn.Linear(40 + action_dim, 30)
        self.l3 = nn.Linear(30, 1)


    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Context module setup as explained previously
        self.context_module = Context(input_dim=state_dim + action_dim + 1, hidden_sizes=[50], output_dim=30, device=device)
        self.actor = Actor(state_dim, action_dim, max_action, self.context_module).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim, self.context_module).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

    def train_context(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done, pre_state, pre_action, pre_reward = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.FloatTensor(reward).to(device)
        not_done = torch.FloatTensor(not_done).to(device)
        pre_info = torch.cat((torch.FloatTensor(pre_action).to(device), 
                              torch.FloatTensor(pre_reward).unsqueeze(-1).to(device),
                              torch.FloatTensor(pre_state).to(device)), dim=-1)

        with torch.no_grad():
            # Prepare context information for next state
            next_pre_info = torch.cat((action, reward.unsqueeze(-1), state), dim=-1)
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, next_pre_info) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, next_action, next_pre_info)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q = self.critic(state, action, pre_info)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state, pre_info), pre_info).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item() if self.total_it % self.policy_freq == 0 else 0}



    def select_action(self, state, last_context):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        last_context = torch.FloatTensor(last_context).to(device)  # Ensure last_context is correctly shaped
        return self.actor(state, last_context).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100):
        for it in range(iterations):
            # Sample replay buffer
            state, action, next_state, reward, not_done, last_context = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            reward = torch.FloatTensor(reward).to(device)
            not_done = torch.FloatTensor(1 - not_done).to(device)
            last_context = torch.FloatTensor(last_context).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state), last_context)
            target_Q = reward + not_done * self.discount * target_Q.detach()

            # Get current Q estimate
            current_Q = self.critic(state, action, last_context)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state, last_context), last_context).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
