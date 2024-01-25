# Model that can be imported to register all tasks
import os
from seqio import FileDataSource, TaskRegistry

from data.metrics import *
from data.preprocessors import *

from config import *


TaskRegistry.add(
  "encoder_only_imagenet2012",
  source=seqio.TfdsDataSource(
    tfds_name="imagenet2012:5.1.0",
    tfds_data_dir=TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
        "label": ["label"]
      }),
    functools.partial(
      encoder_only_preprocessor,
    ),
  ],
  metric_fns=[cls_accuracy_metric],
  output_features=IMAGE_FEATURES,
)

TaskRegistry.add(
  "vitvqgan_imagenet2012",
  source=seqio.TfdsDataSource(
    tfds_name="imagenet2012:5.1.0",
    tfds_data_dir=TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
      }),
    functools.partial(
      vit_vqgan_preprocessor,
    ),
  ],
  metric_fns=[cls_accuracy_metric],
  output_features=VIT_VQGAN_OUTPUT_FEATURES,
)

TFRECORD_LAION400M_FEATURES = {
  'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'text':tf.io.FixedLenFeature(shape=(), dtype=tf.string),
}

TaskRegistry.add(
  "vit_vqgan_liaon_400m",
  source=seqio.TFExampleDataSource(
    split_to_filepattern={
      "train": os.path.join('gs://unified-io-2/pretrain-datasets', "laion400m", "1.0.0", "laion400m-train*"),
    },
    feature_description=TFRECORD_LAION400M_FEATURES,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
      }),
    functools.partial(
      vit_vqgan_preprocessor,
      decode_jpeg=True,
    ),
  ],
  metric_fns=[],
  output_features=VIT_VQGAN_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "vit_vqgan_imagenet2012",
  source=seqio.TfdsDataSource(
    tfds_name="imagenet2012:5.1.0",
    tfds_data_dir=TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
      }),
    functools.partial(
      vit_vqgan_preprocessor,
      decode_jpeg=False,
    ),
  ],
  metric_fns=[],
  output_features=VIT_VQGAN_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "caltech_birds2011",
  source=seqio.TfdsDataSource(
    tfds_name="caltech_birds2011:0.1.1",
    tfds_data_dir=TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image": ["image"],
      }),
    functools.partial(
      vit_vqgan_preprocessor,
      decode_jpeg=False,
    ),
  ],
  metric_fns=[],
  output_features=VIT_VQGAN_OUTPUT_FEATURES,
)

audioset_keys = list(AUDIO_FEATURE_DESCRIPTION.keys())
audioset_keymap = {key: [key] for key in audioset_keys}
audioset_keymap["class_name"] = ["text"]
del audioset_keymap["text"]

TaskRegistry.add(
  "vit_vqgan_audioset",
  source=seqio.TFExampleDataSource(
    split_to_filepattern={
      "train": os.path.join("gs://unified-io-2/pretrain-datasets", "audioset", "1.0.0", "audioset-train*"),
    },
    feature_description=AUDIO_FEATURE_DESCRIPTION,
  ),
  preprocessors=[
        functools.partial(
          rekey, key_map=audioset_keymap,
        ),
        functools.partial(
            audio_preprocessor,
            decode_video_string=True
        ),
    ],
  metric_fns=[],
  output_features=VIT_VQGAN_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "vit_vqgan_acav20m",
  source=seqio.TFExampleDataSource(
    split_to_filepattern={
      "train": os.path.join("gs://unified-io-2/pretrain-datasets", "acav20m_v1", "1.0.0", "acav20m-train*"),
    },
    feature_description=ACAV20M_FEATURE_DESCRIPTION,
  ),
  preprocessors=[
        functools.partial(
          rekey, key_map=audioset_keymap,
        ),
        functools.partial(
            audio_preprocessor,
            decode_video_string=True
        ),
    ],
  metric_fns=[],
  output_features=VIT_VQGAN_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "vit_vqgan_hdvila10m",
  source=seqio.TFExampleDataSource(
    split_to_filepattern={
      "train": os.path.join("gs://unified-io-2/pretrain-datasets", "hdvila10m", "1.0.0", "hdvila10m-train*"),
    },
    feature_description=ACAV20M_FEATURE_DESCRIPTION,
  ),
  preprocessors=[
        functools.partial(
          rekey, key_map=audioset_keymap,
        ),
        functools.partial(
            audio_preprocessor,
            decode_video_string=True
        ),
    ],
  metric_fns=[],
  output_features=VIT_VQGAN_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "vit_vqgan_yttemoporal1b",
  source=seqio.TFExampleDataSource(
    split_to_filepattern={
      "train": os.path.join("gs://unified-io-2/pretrain-datasets", "yttemporal1b", "1.0.0", "yttemporal1b-train*"),
    },
    feature_description=YTTEMOPORAL1B_FEATURE_DESCRIPTION,
  ),
  preprocessors=[
        functools.partial(
          rekey, key_map=audioset_keymap,
        ),
        functools.partial(
            audio_preprocessor,
            decode_video_string=True,
            random_start=False,
        ),
    ],
  metric_fns=[],
  output_features=VIT_VQGAN_OUTPUT_FEATURES,
)
