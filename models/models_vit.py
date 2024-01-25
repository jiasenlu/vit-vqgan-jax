# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Tuple, Type

import flax.linen as nn
from flax.linen.module import merge_param
import jax.numpy as jnp
from jax import lax, random
import numpy as np
import einops

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x

def get_sinusoid_encoding_table(seq_length, emb_dim, dtype):
  """Sinusoid position encoding table: excerpt from original Transformer"""
  def get_position_angle_vec(position):
    return [
      position / np.power(10000, 2 * (dim_j // 2) / emb_dim)
      for dim_j in range(emb_dim)
    ]
  
  sinusoid_table = np.array(
    [get_position_angle_vec(pos_i) for pos_i in range(seq_length)]
  )
  sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
  sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
  
  pos_emb = jnp.array(sinusoid_table).astype(dtype)
  return pos_emb


def drop_path(x: jnp.array, rng, drop_rate: float = 0.) -> jnp.array:
  """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
  This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
  the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
  See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
  changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
  'survival rate' as the argument.
  """
  if drop_rate == 0.:
      return x
  keep_prob = 1. - drop_rate
  mask = random.bernoulli(key=rng, p=keep_prob, shape=(x.shape[0],) + (1,)*(x.ndim-1))
  mask = jnp.broadcast_to(mask, x.shape)
  return lax.select(mask, x / keep_prob, jnp.zeros_like(x))


class DropPath(nn.Module):
  rate: float = 0.
  deterministic: Optional[bool] = None
  
  @nn.compact
  def __call__(self, x, deterministic: bool):
    deterministic = merge_param(
        'deterministic', self.deterministic, deterministic)
    if deterministic or self.rate == 0.:
        return x
    else:
      rng = self.make_rng('drop_path')
    return drop_path(x, rng, self.rate)


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer.
  """

  posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
  dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    """Applies the AddPositionEmbs module.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
    pe = pe.astype(self.dtype)
    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.0
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            inputs)
    x = nn.tanh(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            x)
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.0
  droppath_rate: float = 0.0
  attention_dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)(
            x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = DropPath(rate=self.droppath_rate)(x, deterministic=deterministic) + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)

    return x + DropPath(rate=self.droppath_rate)(y, deterministic=deterministic)


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.0
  droppath_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  add_position_embedding: bool = True

  @nn.compact
  def __call__(self, x, *, train):
    """Applies Transformer model on the inputs.

    Args:
      x: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert x.ndim == 3  # (batch, len, emb)

    if self.add_position_embedding:
      x = AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          dtype=self.dtype,
          name='posembed_input')(
              x)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    else:
      x = get_sinusoid_encoding_table(x.shape[-2], x.shape[-1], self.dtype) + x
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder
    dpr = [x for x in np.linspace(0, self.droppath_rate, self.num_layers)]
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          dtype=self.dtype,
          droppath_rate=dpr[lyr],
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)

    return encoded

def space_to_depth(
    frames: jnp.ndarray,
    spatial_block_size: Any = [1, 1]) -> jnp.ndarray:
  """Space to depth transform."""
  if len(frames.shape) == 4:
    return einops.rearrange(
        frames, 'b (h dh) (w dw) c -> b (h w) (dh dw c)',
        dh=spatial_block_size[0], dw=spatial_block_size[1])
  elif len(frames.shape) == 5:
    return einops.rearrange(
        frames, 'b t (h dh) (w dw) c -> b t (dh dw c)',
        dh=spatial_block_size[0], dw=spatial_block_size[1])
  else:
    raise ValueError(
        'Frames should be of rank 4 (batch, height, width, channels)'
        ' or rank 5 (batch, time, height, width, channels)')

def reverse_space_to_depth(
    frames: jnp.ndarray,
    temporal_block_size: int = 1,
    spatial_block_size: int = 1,
    height: int = 16,
    width: int = 16) -> jnp.ndarray:
  """Reverse space to depth transform."""
  if len(frames.shape) == 3:
    return einops.rearrange(
        frames, 'b (h w) (dh dw c) -> b (h dh) (w dw) c',
        h=height, w=width, dh=spatial_block_size, dw=spatial_block_size)
  elif len(frames.shape) == 4:
    return einops.rearrange(
        frames, 'b h w (dh dw c) -> b (h dh) (w dw) c',
        dh=spatial_block_size, dw=spatial_block_size)
  elif len(frames.shape) == 5:
    return einops.rearrange(
        frames, 'b t h w (dt dh dw c) -> b (t dt) (h dh) (w dw) c',
        dt=temporal_block_size, dh=spatial_block_size, dw=spatial_block_size)
  else:
    raise ValueError(
        'Frames should be of rank 4 (batch, height, width, channels)'
        ' or rank 5 (batch, time, height, width, channels)')

class VisionTransformer(nn.Module):
  """VisionTransformer."""

  num_classes: int
  patch_size: int
  hidden_size: int
  num_layers: int
  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.0
  droppath_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  add_position_embedding: bool = False
  representation_size: Optional[int] = None
  classifier: str = 'token'
  head_bias_init: float = 0.

  @nn.compact
  def __call__(self, inputs, *, train):

    x = inputs
    if x.ndim != 3:
      x = space_to_depth(x, spatial_block_size=self.patch_size)

    x = nn.Dense(
      features=self.hidden_size,
      dtype=self.dtype,
      name='embedding',
    )(x)

    n, l, c = x.shape

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
    
    x = Encoder(
      num_layers=self.num_layers,
      mlp_dim=self.mlp_dim,
      num_heads=self.num_heads,
      dtype=self.dtype,
      dropout_rate=self.dropout_rate,
      droppath_rate=self.droppath_rate,
      attention_dropout_rate=self.attention_dropout_rate,
      add_position_embedding=self.add_position_embedding,
    )(x, train=train)

    if self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    elif self.classifier == 'unpooled':
      pass
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    if self.representation_size is not None:
      x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = IdentityLayer(name='pre_logits')(x)

    if self.num_classes:
      x = nn.Dense(
          features=self.num_classes,
          name='head',
          kernel_init=nn.initializers.zeros,
          bias_init=nn.initializers.constant(self.head_bias_init))(x)
    return x