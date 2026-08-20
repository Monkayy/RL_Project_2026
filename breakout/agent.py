"""Agente DQN: update TD, target network e selezione epsilon-greedy."""

import random
import numpy as np
import torch
from torch import nn

from breakout.config import DQNConfig
from breakout.model import DQN
from breakout.replay_buffer import TransitionBatch


class DQNAgent:
    def __init__(self, n_actions: int, config: DQNConfig, device: torch.device):
        self.config = config
        self.device = device
        self.n_actions = n_actions
        # Initialize online network (actively trained) and target network (stable reference)
        self.online_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        # Target net is used only for inference
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state, epsilon: float) -> int:
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_tensor = torch.as_tensor(np.asarray(state), device=self.device).unsqueeze(0)
            return int(self.online_net(state_tensor).argmax(dim=1).item())

    def update(self, batch: TransitionBatch) -> float:
        # Compute current Q-values for the actions that were actually taken
        current_q = self.online_net(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Select target network for stable bootstrapping
            bootstrap_net = self.target_net if self.config.use_target_network else self.online_net
            next_q = bootstrap_net(batch.next_states).max(dim=1).values

            # Bellman equation target
            targets = batch.rewards + self.config.gamma * (1.0 - batch.dones) * next_q

        loss = self.loss_fn(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def update_target_network(self) -> None:
        # Periodic copy of weights from online -> target network
        self.target_net.load_state_dict(self.online_net.state_dict())