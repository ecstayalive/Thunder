from collections import deque
from typing import Any, Dict

import numpy as np

from .memory import Memory

# TODO: Optimize the speed.
# TODO: Add support for multi-agent simulation environment.


class RolloutBuffer(Memory):
    """Rollout Buffer."""

    def __init__(
        self,
        capacity: int = None,
    ) -> None:
        super(RolloutBuffer, self).__init__()
        self.capacity = capacity
        self.buffer = {
            "observation": deque(maxlen=self.capacity),
            "value": deque(maxlen=self.capacity),
            "action": deque(maxlen=self.capacity),
            "action_log_prob": deque(maxlen=self.capacity),
            "reward": deque(maxlen=self.capacity),
            "next_observation": deque(maxlen=self.capacity),
            "next_value": deque(maxlen=self.capacity),
            "done": deque(maxlen=self.capacity),
        }

    def remember(self, experience: Dict[str, Any]) -> None:
        """To remember experiences.
        Actually for the reply buffer, the experience means some transitions.
        And a transition is presented by a dictionary.

        Args:
            transitions: A dictionary and its keys are
                        ['observation', 'value'
                         'action', 'action_log_prob',
                         'reward', 'next_observation',
                         'next_value', 'done']
        """
        experience_keys = list(experience.keys())
        assert experience_keys == [
            "observation",
            "value",
            "action",
            "action_log_prob",
            "reward",
            "next_observation",
            "next_value",
            "done",
        ], "The transition dict's format is not right, \
            please check the keys of transitions are   \
            'observation', 'action', 'reward', 'next_observation', 'done'"
        for key in experience_keys:
            self.buffer[key].append(experience[key])

    def recall(self, clue: Any) -> Any:
        pass

    def store(self, transitions: Dict[str, Any]) -> None:
        """Store transitions.

        Args:
            transitions: A dictionary and its keys are
                        ['observation', 'value'
                         'action', 'action_log_prob',
                         'reward', 'next_observation',
                         'next_value', 'done']
        """
        self.remember(transitions)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample transitions in buffer.

        Returns:
            sample transitions

        """
        buffer_size = len(self.buffer["done"])
        buffer_keys = self.buffer.keys()
        # Uniforming sampling
        sampling_transitions_indices = np.random.choice(
            buffer_size, batch_size, replace=False
        )
        # Initialize the sampling transitions
        sampling_transitions = {key: self.buffer[key][0] for key in buffer_keys}
        # sample transitions
        for idx in sampling_transitions_indices[1::]:
            for key in buffer_keys:
                sampling_transitions[key] = np.vstack(
                    (sampling_transitions[key], self.buffer[key][idx])
                )

        return sampling_transitions

    def clear(self) -> None:
        """Clear all transitions in transition buffer."""
        for key in self.buffer.keys():
            self.buffer[key].clear()

    def compute_advantage(self, gamma: float, gae_lambda: float) -> None:
        self.buffer["advantage"] = deque(maxlen=self.capacity)
        advantage = 0
        for idx in reversed(range(self.capacity)):
            mask = 1 - self.buffer["done"][idx]
            reward = self.buffer["reward"][idx]
            value = self.buffer["value"][idx]
            next_value = self.buffer["next_value"][idx]
            delta = reward + gamma * next_value * mask - value
            advantage = gamma * gae_lambda * advantage * mask + delta
            self.buffer["advantage"].appendleft(advantage.astype(np.float32))

    @property
    def length(self) -> int:
        return len(self.buffer["done"])
