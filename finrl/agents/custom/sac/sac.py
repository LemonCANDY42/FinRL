# -*- coding: utf-8 -*-
# @Time    : 2023/2/25 16:31
# @Author  : Kenny Zhou
# @FileName: sac.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com

from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import torch as th
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update


SACSelf = TypeVar("SACSelf", bound="SAC")


class CustomSAC(SAC):
  policy_aliases: Dict[str, Type[BasePolicy]] = {
    "MlpPolicy": MlpPolicy,
    "CnnPolicy": CnnPolicy,
    "MultiInputPolicy": MultiInputPolicy,
  }

  def __init__(
      self,
      policy: Union[str, Type[SACPolicy]],
      env: Union[GymEnv, str],
      learning_rate: Union[float, Schedule] = 3e-4,
      buffer_size: int = 1_000_000,  # 1e6
      learning_starts: int = 100,
      batch_size: int = 256,
      tau: float = 0.005,
      gamma: float = 0.99,
      train_freq: Union[int, Tuple[int, str]] = 1,
      gradient_steps: int = 1,
      action_noise: Optional[ActionNoise] = None,
      replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
      replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
      optimize_memory_usage: bool = False,
      ent_coef: Union[str, float] = "auto",
      target_update_interval: int = 1,
      target_entropy: Union[str, float] = "auto",
      use_sde: bool = False,
      sde_sample_freq: int = -1,
      use_sde_at_warmup: bool = False,
      tensorboard_log: Optional[str] = None,
      create_eval_env: bool = False,
      policy_kwargs: Optional[Dict[str, Any]] = None,
      verbose: int = 0,
      seed: Optional[int] = None,
      device: Union[th.device, str] = "auto",
      _init_setup_model: bool = True,
      # (low,high)
      gamma_range: tuple = None,
      # str or list
      gamma_dynamic_mapping_class = "linear"
  ):
    super().__init__(
      policy=policy,
      env=env,
      learning_rate=learning_rate,
      buffer_size=buffer_size,
      learning_starts=learning_starts,
      batch_size=batch_size,
      tau=tau,
      gamma=gamma,
      train_freq=train_freq,
      gradient_steps=gradient_steps,
      action_noise=action_noise,
      replay_buffer_class=replay_buffer_class,
      replay_buffer_kwargs=replay_buffer_kwargs,
      optimize_memory_usage=optimize_memory_usage,
      ent_coef=ent_coef,
      target_update_interval=target_update_interval,
      target_entropy=target_entropy,
      use_sde=use_sde,
      sde_sample_freq=sde_sample_freq,
      use_sde_at_warmup=use_sde_at_warmup,
      tensorboard_log=tensorboard_log,
      create_eval_env=create_eval_env,
      policy_kwargs=policy_kwargs,
      verbose=verbose,
      seed=seed,
      device=device,
      _init_setup_model=_init_setup_model,
    )

    self.gamma_range = gamma_range
    self.gamma_dynamic_mapping_class = gamma_dynamic_mapping_class

    if self.gamma_range:
      if self.gamma_range[0]>=self.gamma_range[-1]:
        raise "gamma_range low must be lower than gamma_range high."
      if self.__gamma<self.gamma_range[0] or self.__gamma>self.gamma_range[-1]:
        raise
      print("Warning:The gamma_range parameter is now enabled, which means the initial gamma value will be replaced by the lower bits of gamma_range.")

      #多段函数
      if len(self.gamma_range)-1 > 1:
        error_string = 'When the gamma_range length is greater than 2. gamma_dynamic_mapping_class should be list or tuple, and the length is equal to the length of gamma_range minus 1.'
        assert(isinstance(self.gamma_dynamic_mapping_class, (list,tuple))), error_string
        if len(self.gamma_range)-1 != len(self.gamma_dynamic_mapping_class):
          raise error_string
      else:
        assert (isinstance(self.gamma_dynamic_mapping_class, (str))), "gamma_dynamic_mapping_class need use string."

      self.__gamma = self.gamma_range[0]
      print("Warning:gamma has been replaced with the lower value in gamma_range.")

    else:
      self.__gamma = gamma
    self.total_timesteps = None

  def __gamma_dynamic_mapping_calculation(self,x,mapping_class:str):
    #重新映射gamma值的mapping函数
    if mapping_class is "square":
      y = x**2
    elif mapping_class is "linear":
      y = x
    else:
      print("Warning: Do you mean \"gamma_dynamic_mapping_class\" is linear?")
      y = x
    return y

  @property
  def gamma(self):
    # 重新映射gamma值
      if self.gamma_range:
        lowest = self.gamma_range[0]
        highest = self.gamma_range[-1]

        #支持分段函数
        for i in range(len(self.gamma_range)-1):

          low  = self.gamma_range[0+i]
          high = self.gamma_range[1+i]

          if self.__gamma>=low and self.__gamma<high:
            if self.num_timesteps <= self.total_timesteps:
              x = self.num_timesteps / self.total_timesteps

              if type(self.gamma_dynamic_mapping_class) is list or type(self.gamma_dynamic_mapping_class) is tuple:
                y = self.__gamma_dynamic_mapping_calculation(x,self.gamma_dynamic_mapping_class[i])
              else:
                y = self.__gamma_dynamic_mapping_calculation(x, self.gamma_dynamic_mapping_class)

              unit = highest-lowest
              new_gamma_value = unit * y + lowest
              # print(f"old gamme:{self.__gamma},new gamme:{new_gamma_value}")
              self.__gamma = new_gamma_value
              return self.__gamma

          else:
            continue

      return self.__gamma

  @gamma.setter
  def gamma(self,new_value):
    self.__gamma = new_value
    print(f"gamma setter {new_value}")

  def learn(
      self: SACSelf,
      total_timesteps: int,
      callback: MaybeCallback = None,
      log_interval: int = 4,
      eval_env: Optional[GymEnv] = None,
      eval_freq: int = -1,
      n_eval_episodes: int = 5,
      tb_log_name: str = "SAC",
      eval_log_path: Optional[str] = None,
      reset_num_timesteps: bool = True,
      progress_bar: bool = False,
  ) -> SACSelf:

    self.total_timesteps = total_timesteps

    return super().learn(
      total_timesteps=total_timesteps,
      callback=callback,
      log_interval=log_interval,
      eval_env=eval_env,
      eval_freq=eval_freq,
      n_eval_episodes=n_eval_episodes,
      tb_log_name=tb_log_name,
      eval_log_path=eval_log_path,
      reset_num_timesteps=reset_num_timesteps,
      progress_bar=progress_bar,
    )

  #Copied from SAC
  def train(self, gradient_steps: int, batch_size: int = 64) -> None:

    super().train(gradient_steps=gradient_steps,batch_size=batch_size)
    self.logger.record("train/gamma", self.__gamma)


