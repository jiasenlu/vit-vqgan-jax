"""
This is the training script
"""

import sys

import os
import yaml
from datetime import datetime
import pytz
import jax
import jax.numpy as jnp
from data.dataloader import input_fn_builder
from training.train_model import *
from flax import jax_utils
from training.optimization import construct_train_state
from models.checkpoint import initialize_using_checkpoint, save_checkpoint, load_checkpoint, bf16_to_f32
from jax.experimental import multihost_utils
import argparse
import numpy as np
import functools
import time
import decimal
import simplejson
from config import plot_spectrogram, eps

# jax.config.update('jax_log_compiles', True)
is_on_gpu = any([x.platform == 'gpu' for x in jax.local_devices()])
if not is_on_gpu:
  assert any([x.platform == 'tpu' for x in jax.local_devices()])
print('JAX process: {} / {}. Local devices {}. Using {}'.format(
  jax.process_index(), jax.process_count(), jax.local_devices(), 'GPU' if is_on_gpu else 'TPU'), flush=True)

parser = argparse.ArgumentParser(description='Train model!')
parser.add_argument(
  'config_file',
  help='Where the config.yaml is located',
  type=str,
)
parser.add_argument(
  '-output_dir',
  help='Override output directory (otherwise we do whats in the config file and add timestamp).',
  dest='output_dir',
  default='',
  type=str,
)

parser.add_argument(
  '-disable_wandb',
  help='dont log this result on weights and biases',
  dest='disable_wandb',
  action='store_true',
)
args = parser.parse_args()

print(f"Loading from {args.config_file}", flush=True)
with open(args.config_file, 'r') as f:
  config = yaml.load(f, yaml.FullLoader)

  seattle_time = pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone('America/Los_Angeles'))
  seattle_time = seattle_time.strftime("%Y-%m-%d-%H:%M.%S")

  if is_on_gpu:
    config['data']['num_train_files'] = 1
    config['device']['output_dir'] = 'temp'
    config['model']['use_bfloat16'] = False
    config['device']['batch_size'] = 6

    config['optimizer']['num_train_steps_override'] = 1000
  elif args.output_dir == '':
    config['device']['output_dir'] = os.path.join(config['device']['output_dir'], seattle_time)
  else:
    config['device']['output_dir'] = args.output_dir

config['_path'] = args.config_file
if (jax.process_index() == 0) and (not is_on_gpu) and (not args.disable_wandb):
  import wandb
  wandb_api, wandb_project, wandb_entity, wandb_name = (
    config['device']['wandb_api'],
    config['device']['wandb_project'],
    config['device']['wandb_entity'],
    config['device']['wandb_name'],
  )
  del config['device']['wandb_api']
  del config['device']['wandb_project']
  del config['device']['wandb_entity']
  del config['device']['wandb_name']
  os.environ["WANDB_API_KEY"] = wandb_api
  wandb.init(
    project=wandb_project,
    entity=wandb_entity,
    name=wandb_name,
    config=config,
  )
else:
  wandb = None


seed = config['device'].get('seed', None)
if seed is None:
  seed = multihost_utils.broadcast_one_to_all(np.int32(time.time()))

ds_train_iter = input_fn_builder(config, seed, is_training=True)

dummy_batch = next(ds_train_iter)

for k, v in dummy_batch.items():
  print("{}: {} {}".format(k, v.shape, v.dtype), flush=True)

aux_rng_keys=["dropout", "drop_path"]

generator = Generator.from_config(config, 'generator')
discriminator = Discriminator.from_config(config, 'discriminator')
lpips = LPIPS.from_config(config, 'vgg')

if is_on_gpu:
  print("DEBUG GPU BATCH!", flush=True)
  rng = jax.random.PRNGKey(0)
  num_keys = len(aux_rng_keys)
  key, *subkeys = jax.random.split(rng, num_keys + 1)
  rng_keys = {aux_rng_keys[ix]: subkeys[ix] for ix in range(len(aux_rng_keys))}
  generator.init({'params': key, **rng_keys}, {k: jnp.asarray(v[0]) for k, v in dummy_batch.items()})

g_params, g_rng_keys = generator.init_from_dummy_batch(dummy_batch, seed, aux_rng_keys)
d_params, d_rng_keys = discriminator.init_from_dummy_batch(dummy_batch, seed, aux_rng_keys)
p_params, p_rng_keys = lpips.init_from_dummy_batch(dummy_batch, seed, aux_rng_keys)

state = construct_train_state(
  opt_config={'g': config['optimizer_g'], 'd': config['optimizer_d']},  
  models={'g': generator, 'd': discriminator, 'p': lpips}, 
  params={'g': g_params, 'd': d_params, 'p': p_params}, 
  rng_keys={'g': g_rng_keys, 'd': d_rng_keys, 'p': p_rng_keys})

step = None
# Initialize params using merlot reserve checkpoint
ckpt_path = config['device'].get('initialize_ckpt', '')
if ckpt_path:
  ckpt = load_checkpoint(path=ckpt_path)
  cache_params_g = ckpt['params_g']
  cache_params_d = ckpt['params_d']
  cache_params_p = ckpt['params_p']
  cache_opt_state_g = ckpt['opt_state_g']
  cache_opt_state_d = ckpt['opt_state_d']
  step = ckpt['step']
  del ckpt

  print(f"{ckpt_path}: {list(cache_params_g.keys())} loaded on the model", flush=True)
  print(f"{ckpt_path}: {list(cache_params_d.keys())} loaded on the model", flush=True)
  print(f"{ckpt_path}: {list(cache_params_p.keys())} loaded on the model", flush=True)

  state = state.replace(
    params_p=initialize_using_checkpoint(state.params_p, cache_params_p),
    params_g=initialize_using_checkpoint(state.params_g, cache_params_g),
    params_d=initialize_using_checkpoint(state.params_d, cache_params_d),
    step = step, 
    )

# load if we can
state = load_checkpoint(state=state, path=config['device']['initialize_ckpt'], step=None,
            use_bfloat16_weights=config['optimizer_g'].get('use_bfloat16_weights', False))
start_step = int(state.step)
state = jax_utils.replicate(state)

p_train_step = jax.pmap(functools.partial(train_step,  config=config,),
             axis_name='batch', donate_argnums=(0, 1,))

  
# p_train_step = jax.vmap(functools.partial(train_step, config=config,),
#                         axis_name='batch')#, donate_argnums=(0, 1,))

train_metrics = []
time_elapsed = []
num_train_steps = config['optimizer_g'].get('num_train_steps_override', config['optimizer_g']['num_train_steps'])
log_every = config['device'].get('commit_every_nsteps', 50)

for n in range(start_step, num_train_steps):
  st = time.time()
  batch = next(ds_train_iter)
  state, loss_info = p_train_step(state, batch)

  # Async transfer. Basically we queue the last thing, then log the thing from `log_every` iterations ago
  if jax.process_index() == 0:
    image_info = {k:v[0] for k, v in loss_info.items() if 'image' in k}
    jax.tree_map(lambda x: x.copy_to_host_async(), image_info)

    loss_info = {k:v for k, v in loss_info.items() if 'image' not in k}
    train_metrics.append(jax.tree_map(lambda x: x[0], loss_info))
    jax.tree_map(lambda x: x.copy_to_host_async(), train_metrics[-1])

    step_for_logging = n - log_every
    if step_for_logging >= 0:
      train_metrics[step_for_logging] = {k: float(v) for k, v in train_metrics[step_for_logging].items()}
      tmp_metrics = {k:v for k, v in train_metrics[step_for_logging].items()}
      if (n + 1) % log_every == 0:
        if wandb is not None:
          for k, v in image_info.items(): tmp_metrics['image' + '/' + k] = wandb.Image(np.array(v[0]), caption=k)
          stats = {
            k: decimal.Decimal("{:.6f}".format(v)) if isinstance(v, float) else v
            for k, v in train_metrics[step_for_logging].items()
          }
          json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
          print("@iter {} stats: {:s}".format(step_for_logging + start_step, json_stats), flush=True)
      if wandb is not None:
        wandb.log(tmp_metrics, step=step_for_logging + start_step, commit=(n + 1) % log_every == 0)

  if (n + 1) % config['device']['save_every_nsteps'] == 0 or (n + 1) == num_train_steps:
    save_checkpoint(state, path=config['device']['output_dir'])
    print(f"Saving @iter {n:03d}.", flush=True)

  time_elapsed.append(time.time() - st)
  if len(time_elapsed) >= 100:
    tsum = sum(time_elapsed)
    print("Completed 100 batches in {:.3f}sec, avg {:.3f} it/sec".format(tsum, 100.0/tsum), flush=True)
    time_elapsed = []

if wandb is not None:
  wandb.finish()