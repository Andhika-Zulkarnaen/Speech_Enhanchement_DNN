import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr

def load_audio(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr)
    return audio / np.max(np.abs(audio))

def save_audio(path, audio, sr=16000):
    sf.write(path, audio, sr)

def spectral_gate(audio, sr=16000, threshold=0.05):
    D = librosa.stft(audio)
    mag = np.abs(D)
    mask = mag > threshold
    return librosa.istft(D * mask)

def reduce_noise(audio, sr=16000):
    return nr.reduce_noise(y=audio, sr=sr, stationary=True)