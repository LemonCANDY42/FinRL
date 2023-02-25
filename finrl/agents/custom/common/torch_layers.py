# -*- coding: utf-8 -*-
# @Time    : 2023/2/25 16:52
# @Author  : Kenny Zhou
# @FileName: torch_layers.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com

import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
from stable_baselines3.sac.policies import SACPolicy

from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)

class CustomLinear(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomLinear, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.flatten = nn.Flatten()
        self.linears = nn.Sequential(
            nn.Linear(n_input_channels, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim//2),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim //4),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            print(self.linears(
                th.as_tensor(self.flatten(observation_space).sample()[None]).float()
            ).shape)

            n_flatten = self.linears(
                th.as_tensor(self.flatten(observation_space).sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.linears(observations))