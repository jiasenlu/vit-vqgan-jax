# Copyright 2022 The T5X Authors.
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

"""Install T5X."""

import os
import sys
import setuptools

# To enable importing version.py directly, we add its path to sys.path.
__version__ = '0.0.0'

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

_jax_version = '0.2.27'
_jaxlib_version = '0.1.76'

setuptools.setup(
    name='vit-vqgan',
    version=__version__,
    description='ViT-VQGAN in JAX',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='AI2',
    author_email='jiasenl@allenai.org',
    url='http://github.com/jiasenlu/vit-vgqan-jax',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    package_data={
        '': ['**/*.gin'],  # not all subdirectories may have __init__.py.
    },
    scripts=[],
    install_requires=[
        'absl-py',
        'cached_property',
        'protobuf==3.19.4',
        'google-api-core==2.8.2',
        # TODO(adarob): Replace with 'clu' once >0.0.6 is released.
        'clu==0.0.8',
        'flax==0.6.3',
        'gin-config',
        f'jax >= {_jax_version},<0.4.0',
        f'jaxlib >= {_jaxlib_version},<0.4.0',
        'numpy',
        'orbax==0.0.2',
        't5',
        'tensorflow',
        'einops',
        'tfds-nightly',
        'tensorflow_probability',
        'tensorflow-addons',
        'tensorflow-datasets @ git+https://github.com/tensorflow/datasets',
        'pycocoevalcap',
        'tensorstore >= 0.1.20',
        'librosa',
        'sk-video',
        'SoundFile',
        'scikit-image',
        'wandb'
    ],
    extras_require={
        'gcp': [
            'gevent', 'google-api-python-client', 'google-compute-engine',
            'google-cloud-storage', 'oauth2client'
        ],
        'test': ['pytest'],

        'data': ['ffmpeg-python', 'scikit-video', 'librosa', 'scikit-image', 'pafy', 'youtube_dl==2020.12.2', 'tensorflow_io', 'pydub'],

        # Cloud TPU requirements.
        'tpu': [f'jax[tpu] >= {_jax_version}'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='machinelearning',
)