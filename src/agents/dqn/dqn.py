from collections import namedtuple, deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from .gnn import GNN


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay

        self.tau = config.tau
        self.total_steps = 0

        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)
        self.policy_net = GNN(edge_index=env.code.data.edge_index).to(config.device)
        self.target_model = GNN(edge_index=env.code.data.edge_index).to(config.device)
        self.target_model.load_state_dict(self.policy_net.state_dict())
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)

        self.criterion = nn.SmoothL1Loss()


    @property
    def epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.total_steps / self.epsilon_decay)


    def select_action(self, state, greedy=False):
        current_epsilon = self.epsilon
        self.total_steps += 1

        if random.random() < current_epsilon and not greedy:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.policy_net(state) # Shape [num_nodes]
                return q_values[self.env.code.q_idx].argmax().item() # Select action with highest Q-value among valid actions (those in q_idx)


    def train_step(self):
        if len(self.replay_buffer) < self.config.batch_size:
            return

        transitions = self.replay_buffer.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self.config.device)
        next_state_batch = torch.stack(batch.next_state).to(self.config.device)
        action_batch = torch.tensor(batch.action, device=self.config.device)
        reward_batch = torch.tensor(batch.reward, device=self.config.device)

        non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch.done)), device=self.config.device, dtype=torch.bool)
        non_final_next_states = next_state_batch[non_final_mask]

        # Q(s_t, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute loss and update policy_net parameters here
        next_q_values = torch.zeros(self.config.batch_size, device=self.config.device)
        with torch.no_grad():
            next_q_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0]


        target_q_values = reward_batch + self.config.gamma * next_q_values
        loss = self.criterion(q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update target network
        target_net_state_dict = self.target_model.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in target_net_state_dict:
            target_net_state_dict[key] = self.tau * policy_net_state_dict[key] + (1 - self.tau) * target_net_state_dict[key]

        self.target_model.load_state_dict(target_net_state_dict)

        return q_values, next_q_values, loss.item()
