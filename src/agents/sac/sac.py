from collections import deque
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch.distributions.categorical import Categorical

from .networks import GNNActor, GNNCritic


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store transition (state: Data, next_state: Data)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(self, env, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.discrete = config.discrete

        self.actor = GNNActor(config, env).to(self.device)
        self.critic1 = GNNCritic(config, env).to(self.device)
        self.critic2 = GNNCritic(config, env).to(self.device)

        # Target networks
        self.target_critic1 = GNNCritic(config, env).to(self.device)
        self.target_critic2 = GNNCritic(config, env).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=config.critic_learning_rate)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=config.critic_learning_rate)

        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)
        self.batch_size = config.batch_size

        self.alpha = config.initial_alpha
        self.log_alpha = torch.tensor(math.log(config.initial_alpha), requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_learning_rate)
        self.target_entropy = -math.log(1.0 / env.code.n_data + 1) * 0.98

        self.gamma = config.gamma
        self.tau = config.tau

        self.total_steps = 0


    def select_action(self, state, evaluate=False):

        if self.discrete:
            with torch.no_grad():
                logits, log_probs, probs = self.actor(state.to(self.device))
                if evaluate:
                    action = probs.argmax(dim=-1)
                else:
                    action = Categorical(probs).sample()
            return action, probs

        else:
             return self.actor(state)[0].detach(), None


    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return {}  # Not enough data to train

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state_batch = Batch.from_data_list(state).to(self.device)
        next_state_batch = Batch.from_data_list(next_state).to(self.device)
        action_batch = torch.tensor(action, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        done_batch = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            logits, log_probs, probs = self.actor(next_state_batch)
            q1_next = self.target_critic1(next_state_batch)  # [B,A]
            q2_next = self.target_critic2(next_state_batch)
            min_q = torch.min(q1_next, q2_next)
            next_v = (probs * (min_q - self.alpha * log_probs)).sum(dim=1, keepdim=True)
            q_target = reward_batch + self.gamma * (1 - done_batch) * next_v
            q_target = torch.clamp(q_target, -10.0, 10.0)

        q1_all = self.critic1(state_batch, action_batch)
        q2_all = self.critic2(state_batch, action_batch)

        q1 = q1_all.gather(1, action_batch.long().unsqueeze(1))
        q2 = q2_all.gather(1, action_batch.long().unsqueeze(1))

        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        critic_loss = q1_loss + q2_loss

        self.critic1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_opt.step()

        logits, log_probs, probs = self.actor(state_batch)

        q1_new = self.critic1(state_batch, logits)
        q2_new = self.critic2(state_batch, logits)
        actor_loss = (probs * (self.alpha * log_probs - torch.min(q1_new, q2_new))).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        entropy = -(probs * log_probs).sum(dim=1).detach()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

        self.total_steps += 1

        train_logs = {
            "Loss/Actor Loss": actor_loss.item(),
            "Loss/Critic Loss": critic_loss.item(),
            "Loss/Alpha Loss": alpha_loss.item(),
            "Monitoring/Alpha": self.alpha,
        }

        return train_logs


    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic1_opt': self.critic1_opt.state_dict(),
            'critic2_opt': self.critic2_opt.state_dict(),
            'alpha_opt': self.alpha_opt.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)