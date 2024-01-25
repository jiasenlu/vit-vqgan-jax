import dataclasses

import jax
from jax import numpy as jnp, lax
import numpy as np
import flax
import flax.linen as nn
from typing import Any, Dict, Union, Optional, Sequence
import clu.parameter_overview
from copy import deepcopy
from dataclasses import dataclass
from models import discriminator

from models.checkpoint import bf16_to_f32
from models.models_vit import VisionTransformer
from models.models_vqgan import Generator
from models.vggish import VGG as audio_vgg
from models.vgg import VGG
from models.discriminator import Discriminator

class Model(nn.Module):
    config: Dict = None
    model_str: str = 'generator'
    
    @classmethod
    def from_config(cls, config, model_str, **kwargs):
        my_config = deepcopy(config)
        my_config['model']['data'] = my_config['data']
        return cls(config=my_config['model'], model_str=model_str, **kwargs)

    def setup(self):
        for k, v in self.config.items():
            setattr(self, k, v)

        self.dtype = jnp.bfloat16 if self.config.get('use_bfloat16', False) else jnp.float32
        print(f"Using dtype {self.dtype}", flush=True)
        
        if self.model_str == 'generator':
            self.model = Generator(
                vocab_size=self.config['vocab_size'],
                proj_dim=self.config['proj_dim'],
                patch_size=self.config['data']['patch_size'],
                encoder_hidden_size=self.config['encoder_hidden_size'],
                encoder_num_layers=self.config['encoder_num_layers'],
                encoder_mlp_dim=self.config['encoder_mlp_dim'],
                encoder_num_heads=self.config['encoder_num_heads'],
                decoder_hidden_size=self.config['decoder_hidden_size'],
                decoder_num_layers=self.config['decoder_num_layers'],
                decoder_mlp_dim=self.config['decoder_mlp_dim'],
                decoder_num_heads=self.config['decoder_num_heads'],
                dtype=self.dtype,
                dropout_rate=self.config.get('dropout_rate', 0.0),
                droppath_rate=self.config.get(' droppath_rate', 0.0),
                attention_dropout_rate=self.config.get('attention_dropout_rate', 0.0),
                add_position_embedding=self.config.get('add_position_embedding', False),
                default_input_size=self.config.get('default_input_size', (256, 256)),
                output_channel=self.config.get('output_channel', 3),
                use_bias=self.config.get('use_bias', False),
                act_fn=self.config.get('act_fn', 'relu'),
            )

        elif self.model_str == 'vgg':
            if self.config['data']['task'] == 'image':
                self.model = VGG(dtype=self.dtype)
            else:
                self.model = audio_vgg(dtype=self.dtype)

        elif self.model_str == 'discriminator':
            if self.config['data']['task'] == 'image':
                self.model = Discriminator(dtype=self.dtype)
            else:
                self.model = Discriminator(
                    num_channels = 1,
                    resolution = 64,
                    dtype=self.dtype, 
                    use_clip=False)
            
    def init_from_dummy_batch(
        self,
        dummy_batch,
        seed=0,
        aux_rng_keys=["dropout", "drop_path"],
    ):

        if self.model_str == 'vgg':
            def init_model(rngs, x0, x1):
                return self.init(rngs, x0, x1, train=False)
        else:
            def init_model(rngs, x):
                return self.init(rngs, x, train=False)
                          
        num_keys = len(aux_rng_keys)
        rng = jax.random.PRNGKey(seed)
        key, *subkeys = jax.random.split(rng, num_keys + 1)
        rng_keys = {aux_rng_keys[ix]: subkeys[ix] for ix in range(len(aux_rng_keys))}
        dummy_batch_jax = {k: jnp.asarray(v[0, 0, None]) for k, v in dummy_batch.items()}
        
        print("start compiling", flush=True)
        x = dummy_batch_jax['inputs']    
        if self.model_str == 'vgg':
            params = jax.jit(init_model, backend='cpu')({'params': key, **rng_keys}, x, x)['params']
            # params = init_model({'params': key, **rng_keys}, x, x)['params']
        else:
            params = jax.jit(init_model, backend='cpu')({'params': key, **rng_keys}, x)['params']
            # params = init_model({'params': key, **rng_keys}, x)['params']

        rngs = flax.core.FrozenDict(rng_keys)

        # in case anything got initialized to bf16
        params = bf16_to_f32(params)
        print(clu.parameter_overview.get_parameter_overview(params), flush=True)

        return params, rngs

    def __call__(self, batch):
        raise NotImplementedError()