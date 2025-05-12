# utils/feature_utils.py

import torch
import torchaudio

def audio_to_features(waveform, n_fft=512, hop_length=128):
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    magnitude = stft.abs()
    return magnitude

def features_to_audio(magnitude, phase, n_fft=512, hop_length=128):
    stft_complex = torch.polar(magnitude, phase)
    waveform = torch.istft(stft_complex, n_fft=n_fft, hop_length=hop_length)
    return waveform
