from models.vggish import VGG
import jax.numpy as jnp
from jax.random import PRNGKey
from training.train_model import *
import yaml

import tensorflow as tf
import seqio
from data.tasks import TaskRegistry
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image

import subprocess
import os
import sys
import librosa
import scipy.signal.windows
import soundfile as sf
import numpy as np
from io import BytesIO
from PIL import Image
from scipy.io import wavfile
import io
from PIL import Image

from models.checkpoint import initialize_using_checkpoint, save_checkpoint, load_checkpoint, bf16_to_f32


# load the data
window_size = 4.08
sample_rate = 16000
n_fft = 1024
win_len = 1024
hop_len=256
n_mels = 128
fmin = 0.0
eps = 0.1
max_wav_value=32768.0
playback_speed = 1
fmax = 8000

audio_fn = 'examples/1aigfM5Tmqk_000020.wav'
sr, waveform = wavfile.read(audio_fn, mmap=True)
waveform = waveform.astype('float32')
waveform /= max_wav_value

st = float(60 * 0 + 0.0)
start_idx = int(sr * st)
end_idx = start_idx + int(sr * window_size) * playback_speed
waveform = waveform[start_idx:end_idx]

librosa_melspec = librosa.feature.melspectrogram(
    waveform,
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_len,
    win_length=win_len,
    center=True,
    pad_mode="reflect",
    power=2.0,
    n_mels=n_mels,
)

audio = librosa_melspec.reshape((1, 1, 128, 256, 1))
audio_mask = tf.cast(audio != 0, tf.float32)
audio = tf.math.log(tf.clip_by_value(audio, 1e-5, 1e5))
audio = (audio + 5.0945) / 3.8312
audio = audio * audio_mask
audio = audio.numpy()

seed = 0
aux_rng_keys=["dropout", "drop_path"]

dummy_batch = {'inputs': audio}

with open('configs/audio_audioset_sh.yaml', 'r') as f:
  config = yaml.load(f, yaml.FullLoader)
generator = Generator.from_config(config, 'generator')

ckpt_path = 'gs://jiasen-us-east/audio_audioset_sh/2022-12-06-11:51.47/ckpt_700000'
ckpt = load_checkpoint(path=ckpt_path)
cache_params_g = ckpt['params_g']
del ckpt

data = audio.reshape(1, 128, 256, 1)

dataset = seqio.get_mixture_or_task("vit_vqgan_audioset").get_dataset(
  sequence_length={},
  split="train",
  num_epochs=1,
  shard_info=seqio.ShardInfo(index=0, num_shards=10),
  use_cached=False,
  seed=42,
  shuffle=False,
)

for ex in zip(dataset.as_numpy_iterator()):
  import pdb; pdb.set_trace()
  data = ex[0]['inputs'].reshape(1, 128, 256, 1)
  rec = generator.apply({'params': cache_params_g},data,train=False,)

