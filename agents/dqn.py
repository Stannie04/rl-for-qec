from collections import deque
import random
import copy
import numpy as np
import torch
from .gnn import GNN
from torch_geometric.data import Batch
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            copy.deepcopy(state),
            action,
            reward,
            copy.deepcopy(next_state),
            done
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return list(states), list(actions), list(rewards), list(next_states), list(dones)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env, **kwargs):
        self.replay_buffer = ReplayBuffer(capacity=kwargs.get("replay_buffer_capacity", 10000))
        self.batch_size = kwargs.get("batch_size", 64)
        self.gamma = kwargs.get("gamma", 0.99)
        self.epsilon_start = kwargs.get("epsilon_start", 1.0)
        self.epsilon_end = kwargs.get("epsilon_end", 0.01)
        self.epsilon_decay = kwargs.get("epsilon_decay", 500)
        self.device = kwargs.get("device", torch.device("cpu"))
        self.target_update_freq = kwargs.get("target_update_freq", 100)
        self.num_timesteps = kwargs.get("num_timesteps", 100000)

        self.env = env
        self.model = GNN().to(self.device)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=kwargs.get("learning_rate", 1e-3))

        self.steps = 0


    def select_action(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.steps / self.epsilon_decay)
        if random.random() < epsilon:
            return self.env.action_space.sample()

        with torch.no_grad():
            q_values = self.model(state)[:self.env.action_space.n]
            return q_values.argmax().item()


    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        state_batch = Batch.from_data_list(states).to(self.device)
        next_state_batch = Batch.from_data_list(next_states).to(self.device)

        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.model(state_batch)  # [total_nodes]

        # Extract per-graph qubit Q-values
        q_values = q_values.view(self.batch_size, -1)[:, :self.env.n_data]
        q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():

            next_q_online = self.model(next_state_batch)
            next_q_online = next_q_online.view(self.batch_size, -1)[:, :self.env.n_data]

            next_actions = torch.argmax(next_q_online, dim=1)

            next_q_target = self.target_model(next_state_batch)
            next_q_target = next_q_target.view(self.batch_size, -1)[:, :self.env.n_data]

            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.steps += 1

        return loss.item()