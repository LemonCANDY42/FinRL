# -*- coding: utf-8 -*-
# @Time    : 2023/2/25 16:34
# @Author  : Kenny Zhou
# @FileName: policies.py
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

from finrl.agents.custom.common.torch_layers import CustomMlpExtractor

class MlpPolicy(SACPolicy):
  """
  Policy class (with both actor and critic) for SAC.

  :param observation_space: Observation space
  :param action_space: Action space
  :param lr_schedule: Learning rate schedule (could be constant)
  :param net_arch: The specification of the policy and value networks.
  :param activation_fn: Activation function
  :param use_sde: Whether to use State Dependent Exploration or not
  :param log_std_init: Initial value for the log standard deviation
  :param sde_net_arch: Network architecture for extracting features
      when using gSDE. If None, the latent features from the policy will be used.
      Pass an empty list to use the states as features.
  :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
      a positive standard deviation (cf paper). It allows to keep variance
      above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
  :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
  :param features_extractor_class: Features extractor to use.
  :param normalize_images: Whether to normalize images or not,
       dividing by 255.0 (True by default)
  :param optimizer_class: The optimizer to use,
      ``th.optim.Adam`` by default
  :param optimizer_kwargs: Additional keyword arguments,
      excluding the learning rate, to pass to the optimizer
  :param n_critics: Number of critic networks to create.
  :param share_features_extractor: Whether to share or not the features extractor
      between the actor and the critic (this saves computation time)
  """

  def __init__(
      self,
      observation_space: gym.spaces.Space,
      action_space: gym.spaces.Space,
      lr_schedule: Schedule,
      net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
      activation_fn: Type[nn.Module] = nn.ReLU,
      use_sde: bool = False,
      log_std_init: float = -3,
      sde_net_arch: Optional[List[int]] = None,
      use_expln: bool = False,
      clip_mean: float = 2.0,
      features_extractor_class: Type[BaseFeaturesExtractor] = CustomMlpExtractor,
      features_extractor_kwargs: Optional[Dict[str, Any]] = None,
      normalize_images: bool = True,
      optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
      optimizer_kwargs: Optional[Dict[str, Any]] = None,
      n_critics: int = 2,
      share_features_extractor: bool = False,
  ):
    super().__init__(
      observation_space,
      action_space,
      lr_schedule,
      net_arch,
      activation_fn,
      use_sde,
      log_std_init,
      sde_net_arch,
      use_expln,
      clip_mean,
      features_extractor_class,
      features_extractor_kwargs,
      normalize_images,
      optimizer_class,
      optimizer_kwargs,
      n_critics,
      share_features_extractor,
    )

  # def _build(self, lr_schedule: Schedule) -> None:
  #     self.actor = self.make_actor(self.features_extractor_class)
  #     self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
  #
  #     if self.share_features_extractor:
  #         self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
  #         # Do not optimize the shared features extractor with the critic loss
  #         # otherwise, there are gradient computation issues
  #         critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
  #     else:
  #         # Create a separate features extractor for the critic
  #         # this requires more memory and computation
  #         self.critic = self.make_critic(features_extractor=None)
  #         critic_parameters = self.critic.parameters()
  #
  #     # Critic target should not share the features extractor with critic
  #     self.critic_target = self.make_critic(features_extractor=None)
  #     self.critic_target.load_state_dict(self.critic.state_dict())
  #
  #     self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)
  #
  #     # Target networks should always be in eval mode
  #     self.critic_target.set_training_mode(False)
