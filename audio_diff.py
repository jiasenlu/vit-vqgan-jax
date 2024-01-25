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
import matplotlib.pyplot as plt
from PIL import Image

sample_rate = 22050
n_fft = 2048
hop_length=736
eps = 0.1

params={
    'n_fft': n_fft,                      # Manually selected by Sangho
    'hop_length': hop_length,            # Manually selected by Sangho
    'window': scipy.signal.windows.hann, # Default
    'n_mels': 64,                        # Manually selected by Sanho
    'fmin': 20.0,                        # Manually selected by Sanho
    'fmax': sample_rate / 2.0,           # Default 22050
}                                    # Spectrogram therefore has shape (64, 376) for 10s

video_fn = '0012y1s1bJI_000350.mp4'
audio_fn = 'audio.wav'
ffmpeg_process = subprocess.Popen(
    ['ffmpeg', '-y', '-i', video_fn, '-ac', '1', '-ar', str(sample_rate), audio_fn],
    stdout=-1, stderr=-1, text=True
)

stdout, stderr = ffmpeg_process.communicate(None, timeout=5.0)
ffmpeg_process.kill()

sr, waveform = wavfile.read(audio_fn, mmap=True)
waveform = waveform.astype('float32')
waveform /= max(np.abs(waveform).max(), 1.0)

window_size = 2.0
playback_speed = 1

st = float(60 * 0 + 0.0)
start_idx = int(sr * st)
end_idx = start_idx + int(sr * window_size) * playback_speed

y = waveform[start_idx:end_idx]

mel = librosa.feature.melspectrogram(y=y, sr=sr, **params)
log_mel = np.log(mel + eps) - np.log(eps)



