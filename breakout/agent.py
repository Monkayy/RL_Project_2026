import random
from typing import Tuple, Optional
import numpy as np
import torch
from torch import nn

from breakout.config import DQNConfig
from breakout.model import DQN, DuelingDQN
from breakout.replay_buffer import TransitionBatch


class DQNAgent:
    def __init__(self, n_actions: int, config: DQNConfig, device: torch.device, seed: int = 0):
        self.config = config
        self.device = device
        self.n_actions = n_actions
        self.rng = random.Random(seed)
        self.eval_rng = random.Random(seed + 1)  
        
        # Initialize online and target network
        model_class = DuelingDQN if config.use_dueling else DQN
        self.online_net = model_class(n_actions).to(device)
        self.target_net = model_class(n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        # Target net is used only for inference
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        # We use none reduction for PER
        self.loss_fn = nn.SmoothL1Loss(reduction='none' if config.use_per else 'mean')

    def select_action(self, state, epsilon: float, evaluating: bool = False) -> int:
        rng = self.eval_rng if evaluating else self.rng
        if rng.random() < epsilon:
            return rng.randrange(self.n_actions)
        with torch.no_grad():
            state_tensor = torch.as_tensor(np.asarray(state), device=self.device).unsqueeze(0)
            return int(self.online_net(state_tensor).argmax(dim=1).item())

    def update(self, batch: TransitionBatch, weights: Optional[torch.Tensor] = None) -> Tuple[float, np.ndarray]:
        # Compute current Q-values for the actions that were actually taken
        current_q = self.online_net(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Select target network for stable bootstrapping
            bootstrap_net = self.target_net if self.config.use_target_network else self.online_net
            
            if self.config.use_double_dqn:
                # DDQN: Online network selects action, target network evaluates it
                next_actions = self.online_net(batch.next_states).argmax(dim=1)
                next_q = bootstrap_net(batch.next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q = bootstrap_net(batch.next_states).max(dim=1).values

            # Bellman equation target
            targets = batch.rewards + self.config.gamma * (1.0 - batch.dones) * next_q

        # Compute TD error for PER
        td_errors = (targets - current_q).abs().detach().cpu().numpy()

        loss = self.loss_fn(current_q, targets)
        
        if self.config.use_per and weights is not None:
            loss = (loss * weights).mean()
        elif self.config.use_per:
            loss = loss.mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return float(loss.item()), td_errors

    def update_target_network(self) -> None:
        # Periodic copy of weights from online -> target network
        self.target_net.load_state_dict(self.online_net.state_dict())