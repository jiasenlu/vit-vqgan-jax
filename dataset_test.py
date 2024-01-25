import tensorflow as tf
import seqio
from data.tasks import TaskRegistry
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image

eps = 0.1
min_level_db=-100

print(tf.executing_eagerly())
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

def _normalize(S):
  return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def plot_spectrogram(data, eps=0.1, n_fft=2048, hop_length=736, sr=22050, fmax=22050/2.0):
    fig, ax = plt.subplots(1, 1)
    data = data * (100) - 100
    data = np.transpose(np.reshape(data, (64, 60)))
    img = librosa.display.specshow(data, x_axis='time', y_axis='mel', sr=sr, fmax=sr/2, n_fft=n_fft, hop_length=hop_length, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


dataset = seqio.get_mixture_or_task("vit_vqgan_yttemoporal1b").get_dataset(
    sequence_length={},
    split="train",
    num_epochs=1,
    shard_info=seqio.ShardInfo(index=0, num_shards=10),
    use_cached=False,
    seed=42,
    shuffle=False,
)
max_value = 0
min_value = 10000

log_mel_list = []
nlog_mel_list = []
n_s_list = []

psum = 0
count = 0
cnt = 0
psum_sq = 0
for ex in zip(dataset.as_numpy_iterator()):
  # img = plot_spectrogram(ex['inputs'])
  # Image.fromarray(img).save('mel_spectrogram.png')
  import pdb; pdb.set_trace()
  psum += ex[0]['inputs'].sum()
  psum_sq += (ex[0]['inputs']**2).sum()

  count += (ex[0]['inputs'] != 0).sum()
  cnt += 1

  if cnt % 1000 == 0:
    print(cnt)

total_mean = psum / count

print(total_mean)
# calulate 

total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = np.sqrt(total_var)  

import pdb; pdb.set_trace()

  # from torchvision.utils import draw_bounding_boxes, save_image
  # import torch
  # import numpy as np
  # image = inputs
  # image_plot = torch.tensor(np.array(image, dtype=np.float32), dtype=torch.float32)
  # image_plot = image_plot.permute(2,0,1)
  # save_image(image_plot, 'input_image.jpg')

#   normalized_log_mel = log_mel / 11.535412
#   spec = np.exp(log_mel + np.log(eps)) - eps
#   S = librosa.power_to_db(spec, ref=np.max)
#   s_normalize = _normalize(S)

#   log_mel_list.append(log_mel)
#   nlog_mel_list.append(normalized_log_mel)
#   n_s_list.append(s_normalize)

# log_mel_list = np.concatenate(log_mel_list, 0)
# nlog_mel_list = np.concatenate(nlog_mel_list, 0)
# n_s_list = np.concatenate(n_s_list, 0)

# log_mel_list = np.reshape(log_mel_list, (-1))
# plt.hist(log_mel_list.tolist(), color = 'blue', edgecolor = 'black')
# plt.savefig('log_mel_list.png')

# plt.clf()
# nlog_mel_list = np.reshape(nlog_mel_list, (-1))
# plt.hist(nlog_mel_list.tolist(), color = 'blue', edgecolor = 'black')
# plt.savefig('nlog_mel_list.png')

# plt.clf()
# n_s_list = np.reshape(n_s_list, (-1))
# plt.hist(n_s_list.tolist(), color = 'blue', edgecolor = 'black')
# plt.savefig('n_s_list.png')

# import pdb; pdb.set_trace()

  # from torchvision.utils import save_image
  # import torch
  # import numpy as np
  # image = x[0]
  # image_plot = torch.tensor(np.array(image, dtype=np.float32), dtype=torch.float32)
  # # image_plot = image_plot.permute(2,0,1)
  # save_image(image_plot, 'input_image1.jpg')
  # import pdb; pdb.set_trace()
