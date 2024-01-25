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

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union, Dict

from flax import struct
import flax.linen as nn
from flax.linen.module import merge_param
import jax
import jax.numpy as jnp
from jax import lax, random
from jax._src import dtypes

import numpy as np
import einops
from models.models_vit import Encoder, space_to_depth, reverse_space_to_depth, DropPath
from jax.nn import initializers
from models.vgg import VGG
from models.discriminator import Discriminator
import collections
from flax.linen.linear import DenseGeneral
from flax.linen.linear import PrecisionLike
from flax.linen.linear import default_kernel_init
from flax.linen.initializers import zeros
from flax.linen.attention import dot_product_attention
import functools

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]

Initializer = Callable[[PRNGKey, Shape, DType], Array]
Dtype = Any

DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex
RealNumeric = Any  # Scalar jnp array or float

KeyArray = random.KeyArray
Array = Any

ACT2FN = {
    "tanh": nn.tanh,
    "relu": nn.relu,
    "swish": nn.swish,
}

def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.
  This function is useful to analyze checkpoints that are without need to access
  the exact source code of the experiment. In particular, it can be used to
  extract an reuse various subtrees of the scheckpoint, e.g. subtree of
  parameters.
  Args:
    keys: a list of keys, where '/' is used as separator between nodes.
    values: a list of leaf values.
  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if '.' not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split('.', 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree


def uniform(scale: RealNumeric = 1e-2, offset: RealNumeric = 0,
            dtype: DTypeLikeInexact = jnp.float_):
  """Builds an initializer that returns real uniformly-distributed random arrays.
  """
  def init(key: KeyArray,
           shape: Shape,
           dtype: DTypeLikeInexact = dtype) -> Array:
    dtype = dtypes.canonicalize_dtype(dtype)
    return random.uniform(key, shape, dtype) * scale + offset
  return init

def l2_normalize(x, axis=None, eps=1e-12):
    """Normalizes along dimension `axis` using an L2 norm.
    This specialized function exists for numerical stability reasons.
    Args:
      x: An input ndarray.
      axis: Dimension along which to normalize, e.g. `1` to separately normalize
        vectors in a batch. Passing `None` views `t` as a flattened vector when
        calculating the norm (equivalent to Frobenius norm).
      eps: Epsilon to avoid dividing by zero.
    Returns:
      An array of the same shape as 'x' L2-normalized along 'axis'.
    """
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)

class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.0
  use_bias: bool = True
  act_fn: str = 'relu'
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
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        )(  # pytype: disable=wrong-arg-types
            inputs)

    x = ACT2FN[self.act_fn](x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        )(  # pytype: disable=wrong-arg-types
            x)
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation
        (default: infer from inputs and params)
      param_dtype: the dtype passed to parameter initializers (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
  """
  num_heads: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: Optional[bool] = None
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  use_bias: bool = True
  attention_fn: Callable[[Array, Array, Array], Array] = dot_product_attention
  decode: bool = False
  params_init: Any = None

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape
        `[batch_sizes..., length, features]`.
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(DenseGeneral,
                              axis=-1,
                              dtype=self.dtype,
                              param_dtype=self.param_dtype,
                              features=(self.num_heads, head_dim),
                              use_bias=False,
                              precision=self.precision)

    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]

    if self.params_init is not None:
      qkv_kernel = jnp.split(jnp.transpose(np.array(self.params_init['to_qkv']['weight']), (1,0)), 3, axis=1)

    query_kernel_init = self.kernel_init if self.params_init is None else lambda *_ : jnp.array(qkv_kernel[0])
    key_kernel_init = self.kernel_init if self.params_init is None else lambda *_ : jnp.array(qkv_kernel[1])
    value_kernel_init = self.kernel_init if self.params_init is None else lambda *_ : jnp.array(qkv_kernel[2])

    query, key, value = (dense(kernel_init=query_kernel_init, name='query')(inputs_q),
                         dense(kernel_init=key_kernel_init, name='key')(inputs_kv),
                         dense(kernel_init=value_kernel_init, name='value')(inputs_kv))

    dropout_rng = None
    if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
      m_deterministic = merge_param('deterministic', self.deterministic,
                                    deterministic)
      if not m_deterministic:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True

    # apply attention
    x = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision)  # pytype: disable=wrong-keyword-args
    # back to the original inputs dimensions

    out_kernel_init = self.kernel_init if self.params_init is None else lambda *_ : jnp.transpose(jnp.array(self.params_init['to_out']['weight']), (1,0))
    out_bias_init = nn.initializers.zeros if self.params_init is None else lambda *_ : jnp.array(self.params_init['to_out']['bias'])

    out = DenseGeneral(features=features,
                       axis=(-2, -1),
                       kernel_init=out_kernel_init,
                       bias_init=out_bias_init,
                       use_bias=self.config.use_bias,
                       dtype=self.dtype,
                       param_dtype=self.param_dtype,
                       precision=self.precision,
                       name='out')(x)
    return out


class TransformerLayer(nn.Module):
  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.0
  droppath_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  use_bias: bool = False
  act_fn: str = 'relu'

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'

    x = nn.LayerNorm(dtype=self.dtype)(inputs)

    # x = MultiHeadDotProductAttention(
    #     dtype=self.dtype,
    #     kernel_init=nn.initializers.xavier_uniform(),
    #     broadcast_dropout=False,
    #     deterministic=deterministic,
    #     dropout_rate=self.attention_dropout_rate,
    #     num_heads=self.num_heads,
    #     use_bias=self.use_bias)(
    #         x, x)

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        use_bias=self.use_bias,
        )(x, x)

    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = DropPath(rate=self.droppath_rate)(x, deterministic=deterministic) + inputs
    
    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)

    y = MlpBlock(
        mlp_dim=self.mlp_dim, 
        dtype=self.dtype, 
        act_fn=self.act_fn,
        dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)

    return x + DropPath(rate=self.droppath_rate)(y, deterministic=deterministic)


class Transformer(nn.Module):
  """Transformer Model for sequence to sequence translation.

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
  use_bias: bool = False
  act_fn: str = 'relu'

  @nn.compact
  def __call__(self, x, *, train):

    assert x.ndim == 3  # (batch, len, emb)

    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    dpr = [x for x in np.linspace(0, self.droppath_rate, self.num_layers)]
    for lyr in range(self.num_layers):
      x = TransformerLayer(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          dtype=self.dtype,
          droppath_rate=dpr[lyr],
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads,
          use_bias=self.use_bias,
          act_fn=self.act_fn,
              )(x, deterministic=not train)

    x = nn.LayerNorm(name='encoder_norm')(x)
    return x


class VectorQuantizer(nn.Module):
  n_e: int
  e_dim: int
  beta: float = 0.25
  embedding_init: Callable[[PRNGKey, Shape, DType], Array] = uniform(2.0, -1.0)
  dtype: Any = jnp.float32
  param_dict: Any = None

  def setup(self):

    kernel_init = self.embedding_init if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['embedding']['weight'])

    self.embedding = self.param(
        'embedding',
        kernel_init, (self.n_e, self.e_dim),
        jnp.float32)

  def get_codebook_entry(self, indices):
    # indices are expected to be of shape (batch, num_tokens)
    # get quantized latent vectors
    z_q = jnp.take(self.embedding, indices, axis=0)
    # normalize latent variable (Ze(x) in the paper)
    z_q = l2_normalize(z_q, axis=-1)
    return z_q
    
  @nn.compact
  def __call__(self, z: Array) -> Array:

    z_reshaped = jnp.reshape(z, (-1, self.e_dim))
    # first normalize the input.
    z_reshaped_norm = l2_normalize(z_reshaped, axis=-1) #/ jnp.linalg.norm(z_reshaped, axis=-1, keepdims=True)
    embedding_norm = l2_normalize(self.embedding, axis=-1) #/ jnp.linalg.norm(self.embedding, axis=-1, keepdims=True)

    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
    d = jnp.sum(z_reshaped_norm ** 2, axis=1, keepdims=True) + \
        jnp.sum(embedding_norm ** 2, axis=1) - 2 * \
        jnp.einsum('ij,kj->ik', z_reshaped_norm, embedding_norm)

    min_encoding_indices = jnp.reshape(jnp.argmin(d, axis=1), z.shape[:-1])

    # z_q = jnp.take(self.embedding, min_encoding_indices, axis=0)
    z_q = self.get_codebook_entry(min_encoding_indices)
    z_norm = l2_normalize(z, axis=-1)

    # e_mean = jnp.mean(min_encoding_indices, axis=0)
    # perplexity = jnp.exp(-jnp.sum(e_mean * jnp.log(e_mean + 1e-10)))
    perplexity = None
    min_encodings = None

    loss = self.beta * jnp.mean(jnp.square((jax.lax.stop_gradient(z_q)-z_norm))) + \
            jnp.mean(jnp.square((z_q - jax.lax.stop_gradient(z_norm))))

    z_q = z + jax.lax.stop_gradient(z_q - z)

    return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

def get_2d_sincos_pos_embed(emb_dim, image_size, image_patch_size, dtype, class_token=False, temperature=10000.):
  """
  (Absolute, additive) 2D sinusoidal positional embeddings used in MoCo v3, MAE
  Args:
    emb_dim (int): embedding dimension
    image_size (tuple): image size
    image_patch_size (int): image patch size
    class_token (bool): whether to use class token
  """
  h, w = image_size[0] // image_patch_size[0], image_size[1] // image_patch_size[1]
  grid_h = jnp.arange(h, dtype=jnp.float32)
  grid_w = jnp.arange(w, dtype=jnp.float32)
  grid_w, grid_h = jnp.meshgrid(grid_w, grid_h, indexing='xy')

  assert emb_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
  emb_w = get_1d_sincos_pos_embed_from_grid(emb_dim // 2, grid_w, jnp.float32, temperature) # (H*W, D/2)
  emb_h = get_1d_sincos_pos_embed_from_grid(emb_dim // 2, grid_h, jnp.float32, temperature) # (H*W, D/2)
  pos_emb = jnp.concatenate([emb_w, emb_h], axis=1) # (H*W, D)
  if class_token:
    pos_emb = jnp.concatenate([jnp.zeros([1, emb_dim], dtype=pos_emb.dtype), pos_emb], axis=0)
  pos_emb = pos_emb.astype(dtype)
  return pos_emb

def get_1d_sincos_pos_embed_from_grid(emb_dim, pos, dtype, temperature=10000.):
  """
  (Absolute, additive) 1D sinusoidal positional embeddings used in MoCo v3, MAE
  Args:
    emb_dim (int):output dimension for each position
    pos: a list of positions to be encoded: size (M, )
    out: (M, D)
  """
  assert emb_dim % 2 == 0
  omega = jnp.arange(emb_dim // 2, dtype=jnp.float32)
  omega /= emb_dim / 2.
  omega = 1. / temperature**omega  # (D/2,)

  pos = pos.reshape(-1).astype(jnp.float32)  # (M,)
  out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

  emb_sin = jnp.sin(out) # (M, D/2)
  emb_cos = jnp.cos(out) # (M, D/2)

  emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
  return emb.astype(dtype)

class Generator(nn.Module):
  """An encoder-decoder Transformer model."""
  vocab_size: int
  proj_dim: int
  patch_size: Any
  encoder_hidden_size: int
  encoder_num_layers: int
  encoder_mlp_dim: int
  encoder_num_heads: int
  decoder_hidden_size: int
  decoder_num_layers: int
  decoder_mlp_dim: int
  decoder_num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.0
  droppath_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  add_position_embedding: bool = False
  head_bias_init: float = 0.
  default_input_size: Any = (256, 256)
  output_channel: int = 3
  use_bias: bool = False
  act_fn: str = 'relu'

  def setup(self):
    self.encoder_position_embedding = get_2d_sincos_pos_embed(
      emb_dim=self.encoder_hidden_size,
      image_size=self.default_input_size,
      image_patch_size=self.patch_size,
      dtype=self.dtype,
      class_token=False,
    )    

    self.decoder_position_embedding = get_2d_sincos_pos_embed(
      emb_dim=self.decoder_hidden_size,
      image_size=self.default_input_size,
      image_patch_size=self.patch_size,
      dtype=self.dtype,
      class_token=False,
    )    

  def encode(self, x, train=True):
    x = space_to_depth(x, spatial_block_size=self.patch_size)
    x = nn.Dense(
        features=self.encoder_hidden_size,
        dtype=self.dtype,
        name='embedding',
        )(x)

    x += jnp.expand_dims(self.encoder_position_embedding, 0)

    x = Transformer(
        num_layers=self.encoder_num_layers,
        mlp_dim=self.encoder_mlp_dim,
        num_heads=self.encoder_num_heads,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        droppath_rate=self.droppath_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        add_position_embedding=self.add_position_embedding,
        use_bias=self.use_bias,
        act_fn=self.act_fn
        )(x, train=train)

    x = ACT2FN[self.act_fn](x)

    x = nn.Dense(
        features=self.proj_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        name='encoder_proj'
        )(x)

    x = nn.LayerNorm(use_scale=False, name='encoder_norm')(x)
    return x 

  def decode(self, x, image_shape, train=True):
    x = nn.Dense(
        features=self.decoder_hidden_size,
        dtype=self.dtype,
        use_bias=self.use_bias,
        name='decoder_proj'
        )(x)

    x += jnp.expand_dims(self.decoder_position_embedding, 0)

    x = Transformer(
        num_layers=self.decoder_num_layers,
        mlp_dim=self.decoder_mlp_dim,
        num_heads=self.decoder_num_heads,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        droppath_rate=self.droppath_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        add_position_embedding=self.add_position_embedding,
        use_bias=self.use_bias,
        act_fn=self.act_fn
        )(x, train=train)

    img_size = self.default_input_size
    x = jnp.reshape(x, (-1, img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1], self.decoder_hidden_size))
  
    x = nn.ConvTranspose(
        features = self.output_channel, 
        kernel_size = self.patch_size, 
        strides = self.patch_size,
        use_bias=self.use_bias,
        )(x)
    return x

  @nn.compact
  def __call__(self, x, *, train):
    
    h = self.encode(x, train=train)
    quant, emb_loss, _ = VectorQuantizer(
        n_e=self.vocab_size,
        e_dim=self.proj_dim,
        beta=0.25,
        )(h)

    rec = self.decode(quant, x.shape[1:], train=train)
    return rec, emb_loss