"""Experience replay a memoria fissa, con frame memorizzati come uint8."""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple[int, ...]):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.states[self.position] = np.asarray(state, dtype=np.uint8)
        self.next_states[self.position] = np.asarray(next_state, dtype=np.uint8)
        self.actions[self.position], self.rewards[self.position] = action, reward
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        if self.size < batch_size:
            raise ValueError("Replay buffer troppo piccolo per il batch richiesto.")
        indices = np.random.randint(0, self.size, size=batch_size)
        return TransitionBatch(
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
        )
