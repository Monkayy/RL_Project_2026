"""Experience replay a memoria fissa e varianti con Prioritized Experience Replay."""

from dataclasses import dataclass
import numpy as np
import torch
from typing import Tuple, Optional


@dataclass
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    indices: Optional[np.ndarray] = None
    weights: Optional[torch.Tensor] = None


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

    def __len__(self) -> int:
        return self.size

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


class SumTree:
    """
    Implementazione di un albero binario in cui il valore
    immagazzinato all'interno di un nodo corrisponde alla
    somma dei valori immagazzinati nei suoi due nodi figli.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0
        self.size = 0

    def add(self, priority: float) -> int:
        tree_index = self.data_pointer + self.capacity - 1
        self.update(tree_index, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return tree_index

    def update(self, tree_index: int, priority: float) -> None:
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v: float) -> Tuple[int, float]:
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        return leaf_index, self.tree[leaf_index]

    @property
    def total_priority(self) -> float:
        return self.tree[0]


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, state_shape: tuple[int, ...], alpha: float = 0.6):
        super().__init__(capacity, state_shape)
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        super().add(state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha)

    def sample(self, batch_size: int, device: torch.device, beta: float = 0.4) -> TransitionBatch:
        if self.tree.size < batch_size:
            raise ValueError("Replay buffer troppo piccolo per il batch richiesto.")

        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            segment_start = segment * i
            segment_end = segment * (i + 1)
            sample_value = np.random.uniform(segment_start, segment_end)

            tree_idx, priority = self.tree.get_leaf(sample_value)
            data_idx = tree_idx - self.tree.capacity + 1
            indices[i] = data_idx
            priorities[i] = priority

        sampling_probabilities = priorities / self.tree.total_priority
        weights = np.power(self.tree.size * sampling_probabilities, -beta)
        weights /= weights.max()

        return TransitionBatch(
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
            indices=indices,
            weights=torch.as_tensor(weights, device=device, dtype=torch.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, td_error in zip(indices, td_errors):
            raw_priority = abs(td_error) + 1e-5
            tree_priority = raw_priority ** self.alpha

            tree_idx = idx + self.tree.capacity - 1
            self.tree.update(tree_idx, tree_priority)

            self.max_priority = max(self.max_priority, raw_priority)
