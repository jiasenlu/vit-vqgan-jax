import seqio
# To ensure all tasks are registered
from data import tasks

MixtureRegistry = seqio.MixtureRegistry

MixtureRegistry.add(
    "audio_datasets",
    [
        "vit_vqgan_audioset",
        "vit_vqgan_acav20m", 
        "vit_vqgan_yttemoporal1b",
    ],
    default_rate=1.0)
