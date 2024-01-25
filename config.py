"""Global UnifiedIO config parameters"""
from typing import Any, Sequence

from flax import struct
from jax import numpy as jnp

import seqio
import tensorflow as tf

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

TFDS_DATA_DIR = 'gs://jiasen-us-east/datasets'

MIN_LEVEL_DB = -100
EPS = 0.1
AMIN = 1e-10
TOP_DB=80.0

# Constants used when encoding region
VOCAB_START = 100
NUM_DETECTION_BIN = 1000

# Controls data augmentation
RANDOM_SCALE_MAX = 1.2
RANDOM_SCALE_MIN = 1.0

# Controls input/output image sizes
IMAGE_INPUT_SIZE = [384, 384]
IMAGE_INPUT_D = 16
IMAGE_TARGET_SIZE = [256, 256]
IMAGE_TARGET_D = 16

FINETUNE_VIDEO_INPUT_SIZE = [384,384]
VIDEO_INPUT_D = 16

IMAGE_FEATURES = {
    "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
    "targets": seqio.ContinuousFeature(dtype=tf.int32, rank=0),
}

VIDEO_FEATURES = {
    "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=3),
    "targets": seqio.ContinuousFeature(dtype=tf.int32, rank=0),
}

VIT_VQGAN_OUTPUT_FEATURES = {
  "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=3),
}

AUDIO_FEATURE_DESCRIPTION = {
    'id': tf.io.FixedLenFeature([], tf.string),
    'text': tf.io.FixedLenFeature([], tf.string),
    'video': tf.io.FixedLenFeature([], tf.string),
    'video_nframes': tf.io.FixedLenFeature([], tf.int64),
    'video_width': tf.io.FixedLenFeature([], tf.int64),
    'video_height': tf.io.FixedLenFeature([], tf.int64),
    'video_nchannels': tf.io.FixedLenFeature([], tf.int64),
    'audio': tf.io.FixedLenFeature([], tf.string),
    'audio_nspectrograms': tf.io.FixedLenFeature([], tf.int64),
    'audio_nmels': tf.io.FixedLenFeature([], tf.int64),
    'audio_nhops': tf.io.FixedLenFeature([], tf.int64)
}

ACAV20M_FEATURE_DESCRIPTION = {
    'id': tf.io.FixedLenFeature([], tf.string),
    'video': tf.io.FixedLenFeature([], tf.string),
    'video_nframes': tf.io.FixedLenFeature([], tf.int64),
    'video_width': tf.io.FixedLenFeature([], tf.int64),
    'video_height': tf.io.FixedLenFeature([], tf.int64),
    'video_nchannels': tf.io.FixedLenFeature([], tf.int64),
    'audio': tf.io.FixedLenFeature([], tf.string),
    'audio_nspectrograms': tf.io.FixedLenFeature([], tf.int64),
    'audio_nmels': tf.io.FixedLenFeature([], tf.int64),
    'audio_nhops': tf.io.FixedLenFeature([], tf.int64)
}

NUM_CHUNKS = 8
YTTEMOPORAL1B_FEATURE_DESCRIPTION = {
    'id': tf.io.FixedLenFeature([], tf.string),
    'video': tf.io.FixedLenFeature([NUM_CHUNKS], tf.string),
    'video_nframes': tf.io.FixedLenFeature([], tf.int64),
    'video_width': tf.io.FixedLenFeature([], tf.int64),
    'video_height': tf.io.FixedLenFeature([], tf.int64),
    'video_nchannels': tf.io.FixedLenFeature([], tf.int64),
    'audio': tf.io.FixedLenFeature([], tf.string),
    'audio_nspectrograms': tf.io.FixedLenFeature([], tf.int64),
    'audio_nmels': tf.io.FixedLenFeature([], tf.int64),
    'audio_nhops': tf.io.FixedLenFeature([], tf.int64),
    'caption': tf.io.FixedLenFeature([NUM_CHUNKS], tf.string),
    'caption_nsentences': tf.io.FixedLenFeature([], tf.int64),
}
eps = 0.1

def plot_spectrogram(log_mel, eps=0.1, ylabel='freq_bin', aspect='auto', xmax=None, to_db=True):
    import librosa
    fig, axs = plt.subplots(1, 1)
    spec = np.exp(log_mel + np.log(eps)) - eps
    if to_db:
        spec = librosa.power_to_db(spec, ref=np.max)
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(spec, origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data