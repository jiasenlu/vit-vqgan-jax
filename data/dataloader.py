"""
Pretraining dataloader
"""
import time
import math
import tensorflow as tf
import seqio
import functools
from copy import deepcopy
import random
import warnings
import numpy as np
import jax
from jax.experimental import multihost_utils
import clu.data

from data.tasks import TaskRegistry
from data.mixtures import MixtureRegistry

from data.data_utils import get_shape_list

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tf.config.experimental.set_visible_devices([], 'GPU')

logger = tf.get_logger()

def handle_batch(batched_tensor, num_devices=None, use_bfloat16=False):
    """
    Deal with the fact that for a batched tensor, the pointers are off
    nvm i'm just not going to worry about that and make the pointers only valid in-batch since we never
    link to anything outside of the batch
    :param batched_tensor:
    :return:
    """
    # Mask batch
    # logger.info("BEFORE HANDLING BATCH")
    # for k, v in batched_tensor.items():
        # logger.info("{}: {}".format(k, v.shape))
    
    batch_size, height, width, channel = get_shape_list(batched_tensor['inputs'], 4)
    if num_devices is not None:
        assert num_devices <= batch_size
        assert batch_size % num_devices == 0
        shape_prefix = [num_devices, batch_size // num_devices]
        # logger.info("{} devices: shape prefix is {}".format(num_devices, shape_prefix))
    else:
        # logger.info("No devices, batch size is just {}".format(batch_size))
        shape_prefix = [batch_size]
    
    batched_tensor["inputs"] = tf.reshape(batched_tensor['inputs'], shape_prefix + [height, width, channel])

    if use_bfloat16:
        batched_tensor['inputs'] = tf.cast(batched_tensor['inputs'], dtype=tf.bfloat16)

    return batched_tensor


def make_dataset(config, batch_size, current_host, num_hosts, num_devices=None, seed=None, is_training=True):
    """
    Create seqio dataset
    :param merged_config:
    :param batch_size:
    :param current_host:
    :param num_hosts:
    :param num_devices:
    :param is_training:
    :return:
    """
    merged_config = deepcopy(config['data'])
    merged_config.update(config['model'])
    if seed is not None:
        multihost_utils.assert_equal(
            np.array(seed),
            f'`seed` is not same across hosts; {jax.process_index()} has a seed of '
            f'{seed}')
        logger.info(
            "Initializing dataset for task '%s' with a replica batch size of %d and "
            'a seed of %d', merged_config['task_name'], batch_size, seed)
    
    mixture_or_task = seqio.get_mixture_or_task(merged_config['task_name'])

    shard_info = seqio.ShardInfo(index=current_host, num_shards=num_hosts)

    sequence_length = {
        'input_size': merged_config['input_size'],
        'patch_size': merged_config['patch_size'],
        'rand_aug': merged_config.get('rand_aug', None),
        'rand_erase': merged_config.get('rand_erase', 0.0),
        'is_training': is_training,
    }

    dataset = mixture_or_task.get_dataset(
        split="train" if is_training else "validation",
        sequence_length=sequence_length,
        shuffle=True if is_training else False,
        shard_info=shard_info,
        trim_output_features=True,
        seed=None,
    )

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(functools.partial(handle_batch, num_devices=num_devices,
                                            use_bfloat16=merged_config['use_bfloat16']))
    return dataset


def input_fn_builder(config, seed, is_training=True, make_dataset_fn=make_dataset):
    """
    Get input fn for TPU use -- for training
    :param config:
    :param is_training:
    :param as_numpy_iter:
    :return:
    """
    import jax
    from flax import jax_utils

    current_host = jax.process_index()
    num_hosts = jax.process_count()
    num_devices = jax.local_device_count()
    batch_size = config['device']['batch_size'] // num_hosts
    random.seed(seed)
    tf.random.set_seed(seed)

    dataset = make_dataset_fn(
        config,
        batch_size=batch_size,
        current_host=current_host,
        num_hosts=num_hosts,
        num_devices=num_devices,
        seed=seed,
        is_training=is_training,
    )

    # dataset = clu.data.TfDatasetIterator(dataset)
    # return dataset

    def _multi_iterator0():
        n_epochs = 0
        while True:
            print(f"Resetting iterator, epoch={n_epochs + 1}", flush=True)
            try:
                dataset_iter = iter(dataset)
                for item in dataset_iter:
                    item = jax.tree_map(lambda x: x._numpy(), item)
                    yield item
            except Exception as e:
                print(str(e))
                time.sleep(5)
            n_epochs += 1

    if config['device'].get('prefetch_size', 1) > 0:
        return jax_utils.prefetch_to_device(_multi_iterator0(), size=config['device'].get('prefetch_size', 1))
    return _multi_iterator0()