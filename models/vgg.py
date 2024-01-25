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
# import t5x.examples.vit_vqgan.layers as layers

URLS = {'vgg16': 'https://www.dropbox.com/s/ew3vhtlg5kks8mz/vgg16_weights.h5?dl=1',
        'vgg19': 'https://www.dropbox.com/s/1sn02fnkj579u1w/vgg19_weights.h5?dl=1'}

def download(ckpt_dir, url):
    name = url[url.rfind('/') + 1 : url.rfind('?')]
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_dir = os.path.join(ckpt_dir, 'flaxmodels')
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        print(f'Downloading: \"{url[:url.rfind("?")]}\" to {ckpt_file}')
        if not os.path.exists(ckpt_dir): 
            os.makedirs(ckpt_dir)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        # first create temp file, in case the download fails
        ckpt_file_temp = os.path.join(ckpt_dir, name + '.temp')
        with open(ckpt_file_temp, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print('An error occured while downloading, please try again.')
            if os.path.exists(ckpt_file_temp):
                os.remove(ckpt_file_temp)
        else:
            # if download was successful, rename the temp file
            os.rename(ckpt_file_temp, ckpt_file)
    return ckpt_file

def normalize_tensor(x, eps=1e-10):
  norm_factor = jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True))  
  return x / (norm_factor + eps)

def spatial_average(x, keepdims=True):
  return jnp.mean(x, axis=[1,2], keepdims=keepdims)

class VGG(nn.Module):
  """
  VGG.
  Attributes:
    output (str):
        Output of the module. Available options are:
            - 'softmax': Output is a softmax tensor of shape [N, 1000] 
            - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
            - 'logits': Output is a tensor of shape [N, 1000]
            - 'activations': Output is a dictionary containing the VGG activations
    pretrained (str):
        Indicates if and what type of weights to load. Options are:
            - 'imagenet': Loads the network parameters trained on ImageNet
            - None: Parameters of the module are initialized randomly
    normalize (bool):
        If True, the input will be normalized with the ImageNet statistics.
    architecture (str):
        Architecture type:
            - 'vgg16'
            - 'vgg19'
    include_head (bool):
        If True, include the three fully-connected layers at the top of the network.
        This option is useful when you want to obtain activations for images whose
        size is different than 224x224.
    num_classes (int):
        Number of classes. Only relevant if 'include_head' is True.
    kernel_init (function):
        A function that takes in a shape and returns a tensor.
    bias_init (function):
        A function that takes in a shape and returns a tensor.
    ckpt_dir (str):
        The directory to which the pretrained weights are downloaded.
        Only relevant if a pretrained model is used. 
        If this argument is None, the weights will be saved to a temp directory.
    dtype (str): Data type.
  """
  output: str='softmax'
  pretrained: str='imagenet'
  normalize: bool=True
  architecture: str='vgg16'
  include_head: bool=False
  num_classes: int=1000
  kernel_init: functools.partial=nn.initializers.lecun_normal()
  bias_init: functools.partial=nn.initializers.zeros
  ckpt_dir: str=None
  lpips: bool=True
  enable_dropout: bool=True
  dropout_rate: float=0.5
  chns: Tuple[int] = (64, 128, 256, 512, 512)
  vgg_output_names: Tuple[str] = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3')
  dtype: str='float32'
  
  def setup(self):
    self.param_dict = None
    if self.pretrained == 'imagenet':
      ckpt_file = download(self.ckpt_dir, URLS[self.architecture])
      self.param_dict = h5py.File(ckpt_file, 'r')

    if self.lpips:  
      self.param_lpips = h5py.File('additional_file/vgg.h5', 'r')

  def _conv_block(self, x, features, num_layers, block_num, act, dtype='float32'):
    
    for l in range(num_layers):
      layer_name = f'conv{block_num}_{l + 1}'
      w = self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict[layer_name]['weight']) 
      b = self.bias_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict[layer_name]['bias']) 

      x = nn.Conv(features=features,
                  kernel_size=(3, 3),
                  padding=((1, 1), (1, 1)),
                  kernel_init=w,
                  use_bias=True,
                  bias_init=b,
                  dtype=self.dtype)(x)
      
      act[layer_name] = x
      x = nn.relu(x)
      act[f'relu{block_num}_{l + 1}'] = x
    return x

  def _net_lin_layer(self, x, block_num, chn_out=1, deterministic=False):
    if self.enable_dropout:
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

    w = self.kernel_init if self.param_lpips is None else \
        lambda *_ : jnp.transpose(jnp.array(self.param_lpips[f'lin{block_num}.model.1.weight']), (2,3,1,0))

    x = nn.Conv(features=chn_out,
                kernel_size=(1,1),
                strides=1,
                padding=((0, 0), (0, 0)),
                kernel_init=w,
                use_bias=False,
                dtype=self.dtype
                    )(x)

    return x
    
  def _forward(self, x):
    act = {}
    x = self._conv_block(x, features=64, num_layers=2, block_num=1, act=act, dtype=self.dtype)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = self._conv_block(x, features=128, num_layers=2, block_num=2, act=act, dtype=self.dtype)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = self._conv_block(x, features=256, num_layers=3 if self.architecture == 'vgg16' else 4, block_num=3, act=act, dtype=self.dtype)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = self._conv_block(x, features=512, num_layers=3 if self.architecture == 'vgg16' else 4, block_num=4, act=act, dtype=self.dtype)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = self._conv_block(x, features=512, num_layers=3 if self.architecture == 'vgg16' else 4, block_num=5, act=act, dtype=self.dtype)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    return x, act

  @nn.compact
  def __call__(self, x0, x1, train=False):
    """
    Args:
      x (tensor of shape [N, H, W, 3]):
          Batch of input images (RGB format). Images must be in range [0, 1].
          If 'include_head' is True, the images must be 224x224.
      train (bool): Training mode.
    Returns:
      If output == 'logits' or output == 'softmax':
          (tensor): Output tensor of shape [N, num_classes].
      If output == 'activations':
          (dict): Dictionary of activations.
    """
    assert x0.shape == x1.shape

    if self.output not in ['softmax', 'log_softmax', 'logits', 'activations']:
      raise ValueError('Wrong argument. Possible choices for output are "softmax", "logits", and "activations".')

    if self.pretrained is not None and self.pretrained != 'imagenet':
      raise ValueError('Wrong argument. Possible choices for pretrained are "imagenet" and None.')

    if self.include_head and (x0.shape[1] != 224 or x0.shape[2] != 224):
      raise ValueError('Wrong argument. If include_head is True, then input image must be of size 224x224.')

    if self.normalize:
      mean = jnp.array([-.030,-.088,-.188]).reshape(1, 1, 1, -1).astype(x0.dtype)
      std = jnp.array([.458,.448,.450]).reshape(1, 1, 1, -1).astype(x0.dtype)

      x0 = (x0 - mean) / std
      x1 = (x1 - mean) / std

    if self.pretrained == 'imagenet':
      if self.num_classes != 1000:
        warnings.warn(f'The user specified parameter \'num_classes\' was set to {self.num_classes} '
                    'but will be overwritten with 1000 to match the specified pretrained checkpoint \'imagenet\', if ', UserWarning)

      num_classes = 1000
    else:
      num_classes = self.num_classes

    x0, act0 = self._forward(x0)
    x1, act1 = self._forward(x1)

    diffs = {}
    # calculate the diff.
    for i, n in enumerate(self.vgg_output_names):
      # normalize the tensor and calculate the diffs.
      diffs[i] = (normalize_tensor(act0[n]) - normalize_tensor(act1[n])) ** 2

    if self.lpips:
      # LPIPS from https://github.com/richzhang/PerceptualSimilarity/blob/31bc1271ae6f13b7e281b9959ac24a5e8f2ed522/lpips/lpips.py#L87
      # spaital average and linear
      res = [spatial_average(self._net_lin_layer(diffs[block_num], block_num, deterministic= not train)) for block_num, _ in enumerate(self.vgg_output_names)]
    else:
      res = [spatial_average(jnp.sum(diffs[block_num], axis=1, keepdims=True), keepdims=True) for block_num, _ in enumerate(self.vgg_output_names)]

    return jnp.reshape(sum(res), (-1))

