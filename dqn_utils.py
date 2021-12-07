import random
import copy
import numpy as np
from collections import namedtuple, deque

import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eps_greedy(agent, action_n, state, eps):
    if random.random() < eps:
        return random.choice(np.arange(action_n))
    return agent.act(state)

class DQN(nn.Module):
    def __init__(self, state_n, action_n, seed):
        super(DQN, self).__init__()
        self.seed = random.seed(seed)
        self.layers = nn.Sequential(nn.Linear(state_n, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_n))

    def forward(self, state):
        return self.layers(state)

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, seed):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, transition):
        e = self.experience(*transition)
        self.buffer.append(e)

    def sample(self):
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, state_n, action_n, gamma, seed, buffer_size=1000, batch_size=128, learning_rate=0.0005):
        self.state_n = state_n
        self.action_n = action_n
        self.batch_size = batch_size
        self.gamma = gamma
        self.EstimateQ = DQN(state_n, action_n, seed)
        self.TargetQ = copy.deepcopy(self.EstimateQ)
        self.buffer = ReplayBuffer(buffer_size, batch_size, seed)
        self.loss = nn.MSELoss()
        self.count = 0
        self.optimizer = torch.optim.Adam(self.EstimateQ.parameters(), lr=learning_rate)

        self.EstimateQ.to(device), self.TargetQ.to(device)

    # state에 대한 argmax action 출력
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.EstimateQ.eval()
        with torch.no_grad():
            Q_value = self.EstimateQ(state)
        self.EstimateQ.train()
        return torch.argmax(Q_value).item()

    # 설정한 batch_size 만큼 모이면 train_batch 생성 후 training 진행
    def step(self, transition):
        self.count = (self.count + 1) % 1000
        self.buffer.add(transition)
        if len(self.buffer) > self.batch_size:
            train_batch = self.buffer.sample()
            self.train(train_batch)
        if self.count == 0:
            self.TargetQ = copy.deepcopy(self.EstimateQ)

    #
    def train(self, train_batch):
        states, actions, rewards, n_states, done = train_batch

        Q_value = self.EstimateQ(states).gather(1, actions)
        Q_prime = torch.max(self.TargetQ(n_states), -1)[0].unsqueeze(1)
        Q_target = rewards + (self.gamma * Q_prime.detach() * (1-done))

        loss = self.loss(Q_value, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path="model.tar.gz"):
        torch.save(self.EstimateQ.state_dict(), path)




