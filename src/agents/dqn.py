from collections import deque
import random
import copy
import numpy as np
from src.experiments.read_config import ConfigParser
from src.environments.qldpc import QLDPCCode
import torch
from .gnn import GNN
from torch_geometric.data import Batch, Data
import torch.nn.functional as F


torch.set_float32_matmul_precision('high')

class ReplayBuffer:
    def __init__(self, capacity=10000, device=torch.device("cpu")):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):

        data = state.clone()
        next_data = next_state.clone()

        self.buffer.append((data, action, reward, next_data, done))

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_batch = Batch.from_data_list(states)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        next_state_batch = Batch.from_data_list(next_states)

        return state_batch, actions, rewards, next_state_batch, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env: QLDPCCode, config: ConfigParser, evaluation_mode: bool=False):
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.device = config.device
        self.target_update_freq = config.target_update_freq

        self.env = env
        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity, device=self.device)

        self.model = GNN().to(self.device)
        self.target_model = copy.deepcopy(self.model)

        self.model = torch.compile(self.model, mode="reduce-overhead")
        self.target_model = torch.compile(self.target_model, mode="reduce-overhead")
        # self.train_step = torch.compile(self.train_step)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.evaluation_mode = evaluation_mode

        self.steps = 0
        self.num_nodes = self.env.data.num_nodes


    def select_action(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.steps / self.epsilon_decay)
        if random.random() < epsilon and not self.evaluation_mode:
            return self.env.action_space.sample()

        with torch.no_grad():
            q_values = self.model(state.x, state.edge_index)[:self.env.action_space.n]
            return q_values.argmax().item()


    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        state_batch, actions, rewards, next_state_batch, dones = self.replay_buffer.sample(self.batch_size)

        q_all = self.model(state_batch.x, state_batch.edge_index)
        q_all = q_all.view(self.batch_size, self.num_nodes, -1)
        q_all = q_all[:, :self.env.n_data, 0]

        q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():

            next_q_online = self.model(next_state_batch.x, next_state_batch.edge_index)
            next_q_online = next_q_online.reshape(self.batch_size, self.num_nodes)
            next_q_online = next_q_online[:, :self.env.n_data]

            next_actions = next_q_online.argmax(dim=1)

            next_q_target = self.target_model(next_state_batch.x, next_state_batch.edge_index)
            next_q_target = next_q_target.reshape(self.batch_size, self.num_nodes)
            next_q_target = next_q_target[:, :self.env.n_data]

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