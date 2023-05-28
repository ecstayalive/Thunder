import os
import pickle
from collections import deque
from typing import Any, Dict

import numpy as np

from .memory import Memory

# TODO: Optimize the speed.
# TODO: Add support for multi-agent simulation environment.


class ReplayBuffer(Memory):
    """Reply Buffer."""

    def __init__(
        self,
        capacity: int = 10000,
        save_buffer: bool = False,
        buffer_file_path: str = None,
    ) -> None:
        if save_buffer and buffer_file_path is None:
            raise ValueError(
                "It seems you are like to save buffer file however you don't give a saving path."
            )
        super(ReplayBuffer, self).__init__()
        self.capacity = capacity
        self.save_buffer = save_buffer
        self.buffer_file_path = buffer_file_path
        self.check_buffer(buffer_file_path)

    def remember(self, experience: Dict[str, Any]) -> None:
        """To remember experiences.
        Actually for the reply buffer, the experience means some transitions.
        And a transition is presented by a dictionary.

        Args:
            experience: transition:['observation', 'action',
                                    'reward', 'next_observation',
                                    'done']

        """
        experience_keys = list(experience.keys())
        assert experience_keys == [
            "observation",
            "action",
            "reward",
            "next_observation",
            "done",
        ], "The transition dict's format is not right, \
            please check the keys of transitions are   \
            'observation', 'action', 'reward', 'next_observation', 'done'"

        buffer_size = len(self.buffer["done"])
        if buffer_size >= self.capacity:
            for key in experience_keys:
                self.buffer[key].popleft()
        for key in experience_keys:
            self.buffer[key].append(experience[key])

    def recall(self, clue: Any) -> Any:
        pass

    def store(self, transitions: Dict[str, Any]) -> None:
        """Store transitions.

        Args:
            transitions: A dictionary and its keys are
                        ['observation', 'action',
                        'reward', 'next_observation',
                        'done']
        """
        self.remember(transitions)

    def save(self) -> None:
        with open(self.buffer_file_path, "wb") as buffer_file:
            pickle.dump(self.buffer, buffer_file)

    def sample(self, batch_size: int = None) -> None:
        """Uniform sample transitions from buffer

        Args:
            batch_size: The sampling size of transitions

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

    def check_buffer(self, buffer_file_path: str) -> None:
        """This function is used to check whether the buffer file exists.
        This function will check if the buffer file exists, and if it is
        existing, load it. And if not, create a buffer.

        Args:
            buffer_file_path: The buffer file path.

        """
        # If given a buffer file save path,
        # first check whether the buffer file exists.
        if buffer_file_path is None:
            self.buffer = {
                "observation": deque(maxlen=self.capacity),
                "action": deque(maxlen=self.capacity),
                "reward": deque(maxlen=self.capacity),
                "next_observation": deque(maxlen=self.capacity),
                "done": deque(maxlen=self.capacity),
            }

        elif os.path.exists(buffer_file_path):
            with open(buffer_file_path, "rb") as buffer_file:
                self.buffer = pickle.load(buffer_file)
            assert list(self.buffer.keys()) == [
                "observation",
                "action",
                "reward",
                "next_observation",
                "done",
            ], "The format of the existing buffer \
                    file is wrong, please delete it."

    @property
    def length(self) -> int:
        return len(self.buffer["done"])
