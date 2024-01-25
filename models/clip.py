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
from flax import struct
from .vgg import download
import collections
from .models_vit import space_to_depth

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]

Initializer = Callable[[PRNGKey, Shape, DType], Array]
Dtype = Any  # this could be a real type?

default_kernel_init = nn.initializers.glorot_uniform()


URLS = {"ViT-B/16": "https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/ViT-B-16.pkl"}

@struct.dataclass
class CLIPConfig:
  image_patch_size: int = 16
  emb_dim: int = 768
  num_heads: int = 12
  num_layers: int = 12
  head_dim: int = 64
  mlp_dim: int = 3072
  mlp_activations: Sequence[str] = ('tanh',)
  dropout_rate: float = 0.0
  float32_attention_logits: bool = False
  default_image_size: int = (256, 256)
  dtype: Any = jnp.bfloat16


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

def QuickGELU(x): return x * nn.sigmoid(1.702 * x)


class MLP(nn.Module):
  config: CLIPConfig
  param_dict: Any = None

  @nn.compact
  def __call__(self, x):
    
    cfg = self.config
    kernel_init = nn.initializers.glorot_uniform() \
        if self.param_dict is None \
        else lambda *_ : jnp.transpose(jnp.array(self.param_dict['c_fc']['weight']), (1,0))
    
    bias_init = nn.initializers.zeros if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['c_fc']['bias'])

    x = DenseGeneral(
      cfg.mlp_dim,
      dtype=cfg.dtype,
      use_bias=True,
      kernel_init=kernel_init,
      bias_init=bias_init,
      name='c_fc',
    )(x)
    
    x = QuickGELU(x)
    
    kernel_init = nn.initializers.glorot_uniform() \
        if self.param_dict is None \
        else lambda *_ : jnp.transpose(jnp.array(self.param_dict['c_proj']['weight']), (1,0))
    
    bias_init = nn.initializers.zeros if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['c_proj']['bias'])

    x = DenseGeneral(
      cfg.emb_dim,
      dtype=cfg.dtype,
      use_bias=True,
      kernel_init=kernel_init,
      bias_init=bias_init,
      name='c_proj',
    )(x)

    return x


def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          dropout_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: DType = jnp.float32,
                          float32_logits: bool = False):
  """Computes dot-product attention given query, key, and value.
  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.
  Args:
    query: queries for calculating attention with shape of `[batch, q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch, kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch, kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.
  Returns:
    Output of shape `[batch, length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # `attn_weights`: [batch, num_heads, q_length, kv_length]
  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)# * depth ** -0.5

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)
  # Normalize the attention weights across `kv_length` dimension.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    dropout_shape = list(attn_weights.shape)
    dropout_shape[-2] = 1
    keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    keep = jnp.broadcast_to(keep, attn_weights.shape)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


dynamic_vector_slice_in_dim = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))

class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.
    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
  """

  num_heads: int
  head_dim: int
  dtype: DType = jnp.float32
  dropout_rate: float = 0.
  kernel_init: Initializer = default_kernel_init
  params_init: Any = None # paramter intialization to pass into the multi-head attention. 
  float32_logits: bool = False  # computes logits in float32 for stability.

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
               abs_bias: Optional[Array] = None,
               *,
               decode: bool = False,
               deterministic: bool = False) -> Array:
    """Applies multi-head dot product attention on the input data.
    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.
    There are two modes: decoding and non-decoding (e.g., training). The mode is
    determined by `decode` argument. For decoding, this method is called twice,
    first to initialize the cache and then for an actual decoding process. The
    two calls are differentiated by the presence of 'cached_key' in the variable
    dict. In the cache initialization stage, the cache variables are initialized
    as zeros and will be filled in the subsequent decoding process.
    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.
    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
      bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
      decode: Whether to prepare and use an autoregressive cache.
      deterministic: Disables dropout if set to True.
    Returns:
      output of shape `[batch, length, q_features]`.
    """

    projection = functools.partial(
        DenseGeneral,
        axis=-1,
        features=(self.num_heads, self.head_dim),
        use_bias=True,
        dtype=self.dtype)

    if self.params_init is not None:
      qkv_kernel = jnp.split(jnp.transpose(np.array(self.params_init['in_proj_weight']), (1,0)), 3, axis=1)
      qkv_bias = jnp.split(np.array(self.params_init['in_proj_bias']), 3, axis=0)

    query_kernel_init = self.kernel_init if self.params_init is None else lambda *_ : jnp.array(qkv_kernel[0])
    query_bias_init = nn.initializers.zeros if self.params_init is None else lambda *_ : jnp.array(qkv_bias[0])

    key_kernel_init = self.kernel_init if self.params_init is None else lambda *_ : jnp.array(qkv_kernel[1])
    key_bias_init = nn.initializers.zeros if self.params_init is None else lambda *_ : jnp.array(qkv_bias[1])

    value_kernel_init = self.kernel_init if self.params_init is None else lambda *_ : jnp.array(qkv_kernel[2])
    value_bias_init = nn.initializers.zeros if self.params_init is None else lambda *_ : jnp.array(qkv_bias[2])

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch, length, num_heads, head_dim]
    query = projection(kernel_init=query_kernel_init, bias_init=query_bias_init, name='query')(inputs_q)
    key = projection(kernel_init=key_kernel_init,  bias_init=key_bias_init, name='key')(inputs_kv)
    value = projection(kernel_init=value_kernel_init, bias_init=value_bias_init, name='value')(inputs_kv)

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None
    
    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, bias, abs_bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
        dtype=self.dtype,
        float32_logits=self.float32_logits)

    out_kernel_init = self.kernel_init if self.params_init is None else lambda *_ : jnp.transpose(jnp.array(self.params_init['out_proj']['weight']), (1,0))
    out_bias_init = nn.initializers.zeros if self.params_init is None else lambda *_ : jnp.array(self.params_init['out_proj']['bias'])
    # Back to the original inputs dimensions.
    out = DenseGeneral(
        features=inputs_q.shape[-1],  # output dim is set to the input dim.
        axis=(-2, -1),
        use_bias=True,
        kernel_init=out_kernel_init,
        bias_init=out_bias_init,
        dtype=self.dtype,
        name='out')(
            x)

    return out

class ResidualAttentionBlock(nn.Module):
  config: CLIPConfig
  param_dict: Any = None

  @nn.compact
  def __call__(self,
               inputs, 
               *,
               enable_dropout: bool = True,
               ):
               
    cfg = self.config
    bias_init = nn.initializers.zeros \
        if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['ln_1']['bias'])

    scale_init = nn.initializers.ones \
      if self.param_dict is None \
      else lambda *_ : jnp.array(self.param_dict['ln_1']['weight'])

    x = nn.LayerNorm(
        epsilon=1e-5,
        bias_init=bias_init, 
        scale_init=scale_init,
        dtype=cfg.dtype, 
        name='ln_1')(inputs)

    x = MultiHeadDotProductAttention(
        num_heads = cfg.num_heads,
        head_dim = cfg.head_dim,
        dtype = cfg.dtype,
        dropout_rate = cfg.dropout_rate,
        params_init = self.param_dict['attn'])(x, x) + inputs

    bias_init = nn.initializers.zeros \
        if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['ln_2']['bias'])

    scale_init = nn.initializers.ones \
      if self.param_dict is None \
      else lambda *_ : jnp.array(self.param_dict['ln_2']['weight'])

    y = nn.LayerNorm(
        epsilon=1e-5,
        bias_init=bias_init, 
        scale_init=scale_init,
        dtype=cfg.dtype, 
        name='ln_2')(x)

    y = MLP(cfg, self.param_dict['mlp'])(y) + x

    return y


class Transformer(nn.Module):
  config: CLIPConfig
  param_dict: Any = None

  @nn.compact
  def __call__(self,
               x, 
               *,
               enable_dropout: bool = True,
               ):
    cfg = self.config

    feat_points = [0, 4, 8]
    x1 = []
    for _ in range(cfg.num_layers):
      x = ResidualAttentionBlock(cfg, param_dict=self.param_dict['resblocks'][str(_)])(x)
      if _ in feat_points:
        x1.append(x)
    
    return x, x1

class VisionTransformer(nn.Module):
  config: CLIPConfig
  param_dict: Any = None

  def setup(self):
    cfg = self.config

    kernel_init = self.embedding_init if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['class_embedding'])
    
    self.class_embedding = self.param(
        'class_embedding',
        kernel_init, 
        (cfg.emb_dim, ),
        jnp.float32)

    kernel_init = self.embedding_init if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['positional_embedding'])
    
    self.positional_embedding = self.param(
        'positional_embedding',
        kernel_init, 
        (197, cfg.emb_dim),
        jnp.float32)

  def get_pos_emb(self):
    
    cls_emb = self.positional_embedding[0:1]
    pos_emb = self.positional_embedding[1:]
    
    pos_emb = jnp.reshape(pos_emb, 
        (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1]))

    pos_emb = jax.image.resize(pos_emb, 
        (16, 16, pos_emb.shape[-1]), "bicubic")
    
    pos_emb = jnp.reshape(pos_emb, [-1, pos_emb.shape[-1]])
    
    return jnp.concatenate([cls_emb, pos_emb], axis=0)[None,:,:]


  @nn.compact
  def __call__(self,
               x, 
               *,
               enable_dropout: bool = True,
               ):
    cfg = self.config

    B = x.shape[0]
    x = space_to_depth(x, spatial_block_size=[cfg.image_patch_size,cfg.image_patch_size])

    # get the initialization here.
    kernel_init = nn.initializers.glorot_uniform() \
        if self.param_dict is None else lambda *_ : jnp.reshape(jnp.transpose(jnp.array(self.param_dict['conv1']['weight']), (2,3,1,0)), (-1, cfg.emb_dim))

    x = DenseGeneral(
        features=cfg.emb_dim,
        use_bias=False,
        kernel_init=kernel_init,
        dtype=cfg.dtype,
        # precision=jax.lax.Precision('highest'),
        name='embedding')(
            x)

    x = jnp.concatenate([
        jnp.repeat(self.class_embedding[None, None, :], B, axis=0),
        x], axis=1)

    x = x + self.get_pos_emb() #self.positional_embedding
    # x = x + self.positional_embedding
    bias_init = nn.initializers.zeros \
        if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['ln_pre']['bias'])

    scale_init = nn.initializers.ones \
      if self.param_dict is None \
      else lambda *_ : jnp.array(self.param_dict['ln_pre']['weight'])

    x = nn.LayerNorm(
        epsilon=1e-5,
        bias_init=bias_init, 
        scale_init=scale_init,
        dtype=cfg.dtype, 
        name='pre_ln')(x)

    x, x1 = Transformer(cfg, param_dict=self.param_dict['transformer'])(x)


    bias_init = nn.initializers.zeros \
        if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['ln_post']['bias'])

    scale_init = nn.initializers.ones \
      if self.param_dict is None \
      else lambda *_ : jnp.array(self.param_dict['ln_post']['weight'])
        
    x = nn.LayerNorm(
        epsilon=1e-5,
        bias_init=bias_init, 
        scale_init=scale_init,
        dtype=cfg.dtype, 
        name='pre_post')(x[:,0,:])
    
    kernel_init = nn.initializers.glorot_uniform() if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['proj'])

    x = DenseGeneral(
        features=512,
        use_bias=False,
        kernel_init=kernel_init,
        dtype=cfg.dtype,
        name='proj')(
            x)

    x1.append(x)
    return x1

class BlurPool(nn.Module):
  pad_type: str = 'zero' 
  filt_size: int = 4
  stride: int = 1
  pad_off: int = 0
  dtype: Any = jnp.float32

  def setup(self):
  
    self.pad_sizes = int(1.*(self.filt_size-1)/2) + self.pad_off

    if(self.filt_size==1):
      a = np.array([1.,])
    elif(self.filt_size==2):
      a = np.array([1., 1.])
    elif(self.filt_size==3):
      a = np.array([1., 2., 1.])
    elif(self.filt_size==4):    
      a = np.array([1., 3., 3., 1.])
    elif(self.filt_size==5):    
      a = np.array([1., 4., 6., 4., 1.])
    elif(self.filt_size==6):    
      a = np.array([1., 5., 10., 10., 5., 1.])
    elif(self.filt_size==7):    
      a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = a[:,None]*a[None,:]
    filt = filt / np.sum(filt)
    filt = np.tile(filt[None, None, :, :], [256, 1, 1, 1])
    
    self.filt = jnp.array(filt, dtype=self.dtype)

  @nn.compact
  def __call__(self, x, *, train: bool = True):

    x = jnp.transpose(x, (0, 3, 1, 2))
    x = jnp.pad(x, pad_width=((0,0),(0,0),(self.pad_sizes,self.pad_sizes),(self.pad_sizes,self.pad_sizes)))
    x = jax.lax.conv_general_dilated(
        x, self.filt, (self.stride, self.stride), padding='valid', feature_group_count=x.shape[1])
    x = jnp.transpose(x, (0,2,3,1))
    return x

class MultiLevelDViT(nn.Module):
  level: int = 4
  out_ch: int = 256
  num_classes: int = 0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, xs, *, train: bool = True):
  
    final_pred = []
    for i in range(self.level-1):
      x = xs[i][:,1:,:]
      x = jnp.reshape(x, [x.shape[0], 16, 16, x.shape[-1]])
      x = nn.Conv(
          features=self.out_ch,
          kernel_size=(3, 3),
          strides=2,
          padding=((0, 0), (0, 0)),
          dtype=self.dtype,
          )(x)

      x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(x)
      x = nn.activation.leaky_relu(x, negative_slope=0.2)

      x = nn.Conv(
          features=self.out_ch,
          kernel_size=(3, 3),
          strides=2,
          padding=((0, 0), (0, 0)),
          dtype=self.dtype,
          )(x)

      x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(x)
      x = nn.activation.leaky_relu(x, negative_slope=0.2)

      x = BlurPool(pad_type='zero', stride=1, dtype=self.dtype)(x)
      x = nn.Conv(
          features=1,
          kernel_size=(1, 1),
          strides=2,
          padding='valid',
          dtype=self.dtype,
          )(x)
      # x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(x)
      final_pred.append(x)

    x = nn.Dense(self.out_ch)(xs[-1])
    x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(x)
    x = nn.activation.leaky_relu(x, negative_slope=0.2)
    x = nn.Dense(1)(x)
    final_pred.append(x)

    return final_pred



class CLIP(nn.Module):
  """An encoder-decoder Transformer model."""
  config: CLIPConfig

  def setup(self):
    cfg = self.config

    # load the pre-trained weight of ViT-B/16.
    ckpt_file = download(None, URLS["ViT-B/16"])
    ckpt_dict = pickle.load(open(ckpt_file, 'rb'))
    keys, values = zip(*list(ckpt_dict.items()))    
    self.param_dict = recover_tree(keys, values)

    self.image_mean = jnp.array([0.48145466, 0.4578275, 0.40821073], self.config.dtype)
    self.image_std = jnp.array([0.26862954, 0.26130258, 0.27577711], self.config.dtype)

  @nn.compact
  def __call__(self,
               x, 
               *,
               train: bool = True,
               ):
    

    cfg = self.config
    x = x - self.image_mean[None, None, None, :]
    x /= self.image_std[None, None, None, :]

    x = VisionTransformer(
          config = cfg,
          param_dict = self.param_dict['visual']
          )(x)

    return x