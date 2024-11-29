import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


# Deep Q-Network
class DQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


# DQN Agent
class DQNAgent:
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims,
                 epsilon_dec=0.996, epsilon_end=0.01, mem_size=10000, replace_target=1000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_end
        self.epsilon_dec = epsilon_dec
        self.lr = alpha
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DQNetwork(input_dims, n_actions)
        self.q_next = DQNetwork(input_dims, n_actions)

        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.q_next.load_state_dict(self.q_eval.state_dict())
        self.q_next.eval()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            actions = self.q_eval(state)
            return torch.argmax(actions).item()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.optimizer.zero_grad()

        states, actions, rewards, states_, terminal = self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_ = torch.tensor(states_, dtype=torch.float32)
        terminal = torch.tensor(terminal, dtype=torch.float32)

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval(states)[indices, actions]
        q_next = self.q_next(states_).max(dim=1)[0]
        q_target = rewards + self.gamma * q_next * terminal

        loss = self.loss(q_pred, q_target)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.replace_target == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

    def save_model(self, filename):
        torch.save(self.q_eval.state_dict(), filename)

    def load_model(self, filename):
        self.q_eval.load_state_dict(torch.load(filename))
        self.q_next.load_state_dict(self.q_eval.state_dict())
