import random
from functools import reduce
import einops

from config import *
from data.data_utils import *
from data.imagenet_utils import _preprocess_image, RandomErasing


AUTOTUNE = tf.data.experimental.AUTOTUNE
rekey = seqio.preprocessors.rekey

def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def get_from_dict(data, keys):
    """Iterate nested dictionary"""
    return reduce(dict.get, keys, data)

@seqio.utils.map_over_dataset
def rekey(x, key_map=None):
  """Replace the feature keys according to the mapping in `key_map`.
  For example, if the dataset returns examples of the format:
  {'foo': 'something', 'bar': 'something else'}
  and key_map = {'boo': 'foo', 'spar': 'bar'} then this function will return
  examples with the format
  {'boo': 'something', 'spar': 'something else'}
  If a mapping is to an empty key or None, set the new key to an empty string.
  Args:
      x: an example to process.
      key_map: dictionary mapping new keys to original keys
  Returns:
      A preprocessed example with the format listed above.
  """
  if key_map:
    return {
        new_key: get_from_dict(x, old_key) if old_key else ''
        for new_key, old_key in key_map.items()
    }
  return x


def vit_vqgan_preprocessor(ds, sequence_length, decode_jpeg=False):

  image_input_size = [256, 256]
  is_training = sequence_length.get('is_training', True)

  def to_inputs_and_targets(ex):
    if decode_jpeg:
      img = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      img = ex['image']

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                    do_random_scale=is_training,
                                                    random_scale_max=RANDOM_SCALE_MAX,
                                                    random_scale_min=RANDOM_SCALE_MIN,
                                                    shrink_both_sides=True,
                                                    do_flip_if_vertical=False,
                                                    random_scale_ratio=1)

    image_info, masks, boxes, labels, indices = this_image_info
    
    #[-1, 1]
    img = img * 2 - 1 
    return {'inputs': img}
    
  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def audio_preprocessor(ds, sequence_length, decode_video_string=True, random_start=True):
  
  def to_inputs_and_targets(ex):
    audio_shape = (ex['audio_nspectrograms'], ex['audio_nmels'], ex['audio_nhops'])
    spectrograms = tf.io.parse_tensor(ex['audio'],  tf.float32)
    spectrograms = tf.reshape(spectrograms, audio_shape)

    rand_idx = tf.random.uniform([], minval=0, maxval=tf.shape(spectrograms)[0], dtype=tf.int32)    
    audio = spectrograms[rand_idx]
    # random augment the audio masks. for 2 second audio, it will [72, 192]
    
    if random_start:
      audio = audio[:, 72:192]
      # random select a start point. 
      rand_start = tf.random.uniform([], minval=0, maxval=17, dtype=tf.int32)
      audio_padded = tf.pad(audio, [[0,0], [rand_start*8, (17-rand_start)*8]], "CONSTANT")
      audio_mask = tf.cast(audio_padded != 0, tf.float32)
    else:
      audio_padded = audio
      audio_mask = tf.ones_like(audio_padded)

    audio_padded = tf.math.log(tf.clip_by_value(audio_padded, 1e-5, 1e5))
    audio_padded = (audio_padded + 5.0945) / 3.8312
    audio_padded = audio_padded * audio_mask
    audio_padded = tf.expand_dims(audio_padded, -1)
    
    return {'inputs': audio_padded}

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def encoder_only_preprocessor(
  ds, sequence_length, class_id=None, class_map=None, decode_jpeg=False,
):
  image_input_size = sequence_length.get('input_size', IMAGE_INPUT_SIZE)
  image_input_d = sequence_length.get('patch_size', IMAGE_INPUT_D)
  is_training = sequence_length.get('is_training', True)
  rand_aug = sequence_length.get('rand_aug', None)
  rand_erase = sequence_length.get('rand_erase', 0.0)
  if rand_aug is not None:
    augmentation_settings = {}
    augmentation_settings['randaugment'] = dict(num_layers=rand_aug[0], magnitude=rand_aug[1])
    augmentation_settings['cutmix'] = False
    augmentation_settings['mixup_alpha'] = None
  
  if class_id is not None:
    keys_tensor = tf.constant(class_id)
    table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
        keys_tensor,
        tf.constant([i for i in range(len(class_id))], tf.int32),
      ),
      default_value=-1,
    )
  elif class_map is not None:
    keys_tensor = tf.constant([i for i in range(len(class_map))], tf.int32)
    table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
        keys_tensor,
        class_map
      ),
      default_value=21843,
    )
  if rand_erase > 0:
    random_eraser = RandomErasing(probability=rand_erase)

  def to_inputs_and_targets(ex):
    if rand_aug is not None:
      image_inputs, _ = _preprocess_image(
        ex['image'], is_training, image_input_size, augmentation_settings)
      image_input_masks = tf.ones_like(image_inputs)[:, :, 0]
    else:
      if decode_jpeg:
        img = tf.image.decode_jpeg(ex['image'], channels=3)
      else:
        img = ex['image']

      img = tf.image.convert_image_dtype(img, dtype=tf.float32)
      img, img_mask, this_image_info = resize_and_pad(img, image_input_size,
                                                      do_random_scale=is_training,
                                                      random_scale_max=RANDOM_SCALE_MAX,
                                                      random_scale_min=RANDOM_SCALE_MIN,
                                                      shrink_both_sides=True,
                                                      do_flip_if_vertical=False,
                                                      random_scale_ratio=0.5,
                                                      resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)
      image_inputs = img
      image_inputs = normalize_image(image_inputs)
      image_input_masks = img_mask
    if is_training and rand_erase > 0:
      image_inputs = random_eraser.distort(image_inputs)
    
    # Arrange into a list of patches
    image_inputs = einops.rearrange(
      image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
      dh=image_input_d, dw=image_input_d)
    
    if 'class_id' in ex:
      cid = ex['class_id']
      label = table.lookup(cid)
      ex['label'] = label
    elif class_map is not None:
      ex['label'] = table.lookup(tf.cast(ex['label'], tf.int32))
    return {
      'inputs': image_inputs,
      'targets': tf.cast(ex['label'], tf.int32),
    }
  
  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def video_encoder_only_preprocessor(ds, sequence_length, class_name=None, decode_video_string=False):
    video_input_size = sequence_length.get('input_size', FINETUNE_VIDEO_INPUT_SIZE)
    video_input_d = sequence_length.get('patch_size', VIDEO_INPUT_D)
    num_frames = sequence_length.get('num_frames', 4)

    if class_name is not None:
        table = create_lookup_table_of_class_names(class_name)

    is_training = sequence_length.get('is_training', True)

    def to_inputs_and_targets(ex):

        # create video_inputs shape: (T,H,W,C) [0,255]
        if decode_video_string:
            # parse if stored as a TFRecord (byte string)
            video_shape = (ex['video_nframes'], ex['video_height'], ex['video_width'], ex['video_nchannels'])
            video = tf.io.parse_tensor(ex['video'], tf.uint8)
            video = tf.reshape(video, video_shape)
            # save shape in the TFRecord
            # tf.reshape(video, shape)
        else:
            video = ex['video']

        video = convert_video_dtype(video,tf.float32) # [0,1]
        #video: TxHxWx3; video_mask: TxHxW

        video, video_mask, _ = resize_and_pad(
            video, 
            is_video=True,
            desired_output_size=video_input_size,
            do_random_scale=is_training,
            random_scale_max=RANDOM_SCALE_MAX,
            random_scale_min=RANDOM_SCALE_MIN,
            shrink_both_sides=True,
            do_flip_if_vertical=False,
            random_scale_ratio=0.5,
            resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)
        video_inputs = video
        video_input_masks = video_mask

        # Sample a fixed number of frames
        # [T, H, W, C]
        video_inputs, indices = sample_uniform_sequence(
            sequence=video_inputs,
            num_steps=num_frames,
            random=is_training,
        )
        video_inputs = normalize_video(video_inputs)

        video_input_masks = tf.gather(video_input_masks, indices)

        video_inputs = einops.rearrange(
            video_inputs, 't (h dh) (w dw) c -> t (h w) (dh dw c)',
            dh=video_input_d, dw=video_input_d)

        # create text_targets
        label = tf.cast(ex['label'], tf.int32)
        
        return {
            'inputs': video_inputs,
            'targets': label,
        }

    return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)