# -*- coding: utf-8 -*-
# @Time    : 2023/2/25 16:31
# @Author  : Kenny Zhou
# @FileName: sac.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.sac.policies import CnnPolicy, MultiInputPolicy
from finrl.agents.custom.sac.policies import MlpPolicy

class CustomSAC(SAC):

  policy_aliases: Dict[str, Type[BasePolicy]] = {
    "MlpPolicy": MlpPolicy,
    "CnnPolicy": CnnPolicy,
    "MultiInputPolicy": MultiInputPolicy,
  }
