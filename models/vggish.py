from jax import random
import jax.numpy as jnp
import flax.linen as nn
import functools
from typing import Any, Tuple
import h5py
import warnings

from tqdm import tqdm
import requests
import os
import tempfile

from models.vgg import download

URLS = {
  'vggish': 'https://github.com/harritaylor/torchvggish/'
            'releases/download/v0.1/vggish-10086976.pth',
  'pca': 'https://github.com/harritaylor/torchvggish/'
          'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}

def normalize_tensor(x, eps=1e-10):
  norm_factor = jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True))  
  return x / (norm_factor + eps)

def spatial_average(x, keepdims=True):
  return jnp.mean(x, axis=[1,2], keepdims=keepdims)

class VGG(nn.Module):
  dtype: Any = jnp.float32
  ckpt_dir: str = None
  
  def setup(self):
    ckpt_file = download(self.ckpt_dir, URLS['vggish'])
    import torch
    self.param_dict = torch.load(ckpt_file)
  
  def _forward(self, x):
    out = []
    cnt = 0
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
      if v == "M":
        out.append(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        cnt += 1
      else:
        w = lambda *_ : jnp.transpose(jnp.array(self.param_dict[f'features.{cnt}.weight']), (2,3,1,0))
        b = lambda *_ : jnp.array(self.param_dict[f'features.{cnt}.bias'])      
        x = nn.Conv(
            features=v, 
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            kernel_init=w,
            use_bias=True,
            bias_init=b,
            dtype=self.dtype)(x)
        cnt += 1

        x = nn.relu(x)
        cnt += 1

    return out

  @nn.compact
  def __call__(self, x0, x1, train=False):
    
    act0 = self._forward(x0)
    act1 = self._forward(x1)

    diffs = {}

    num = len(act0)
    for i in range(num):
      diffs[i] = (normalize_tensor(act0[i]) - normalize_tensor(act1[i])) ** 2

    res = [spatial_average(jnp.sum(diffs[i], axis=-1, keepdims=True), keepdims=True) for i in range(num)]
    
    return jnp.reshape(sum(res), (-1))  
