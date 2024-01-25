#!/usr/bin/env bash
set -e

# This script will get ran on the servers

# this locks the python executable down to hopefully stop if from being fiddled with...
screen -d -m python -c 'import time; time.sleep(999999999)'

# initializes jax and installs ray on cloud TPUs.
# Note that `clu` already installs tensorflow-cpu, which is needed to kick out the default tensorflow
# installing on sudo doesn't work, idk why though
/usr/bin/python3 -m pip install --upgrade pip

cd ~

python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install --upgrade fabric dataclasses optax tqdm cloudpickle smart_open[gcs] func_timeout aioredis==1.3.1 wandb pandas simplejson

# 32 * 1024 ** 3 -> 32 gigabytes
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=34359738368