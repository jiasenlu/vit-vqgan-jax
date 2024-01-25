
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
import math
import functools

from flax.linen.linear import DenseGeneral
import flax.linen as nn
from flax.linen.module import merge_param

import jax
import jax.numpy as jnp
from jax import lax, random
import numpy as np
import einops
import re
import pickle

from models.clip import CLIP, CLIPConfig, MultiLevelDViT
from models.stylegan_discriminator import stylegan_discriminator

class Discriminator(nn.Module):
  """Discriminators"""
  dtype: Any = jnp.float32
  num_channels: int = 3
  resolution: int = 256
  use_clip: bool = True

  def setup(self):
    self.stylegan_disc = stylegan_discriminator(
        num_channels = self.num_channels,
        dtype = self.dtype,
        resolution = self.resolution)

    if self.use_clip:
      self.clip = CLIP(CLIPConfig)
      self.multi_level_dvit = MultiLevelDViT(dtype=self.dtype)

  def get_stylegan_logit(self, x, train=True):
    return self.stylegan_disc(x, train=train)

  def get_clip_feature(self, x, train=False):
    return self.clip(x, train=False)
  
  def get_clip_logit(self, x, train=True):
    return self.multi_level_dvit(x, train=train)

  @nn.compact
  def __call__(self, x, c=None, train=True): 
    stylegan_logit = self.get_stylegan_logit(x, train=train)

    clip_logit = None
    if self.use_clip:
      clip_feat = self.get_clip_feature(x, train=train)
      clip_logit = self.get_clip_logit(clip_feat, train=train)

    return stylegan_logit, clip_logit