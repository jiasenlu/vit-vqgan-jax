# VIT-VQGAN

## Run code on TPU VMs

Once you've created a TPU VM, run

```bash
python tpu_run.py
```

Basically this SSH's into your TPU VM, installs dependencies, and then runs your command.
You should modify the code in `tpu_run.py` and configuration in `pretrain/configs/*.yaml`.