import os
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf
import librosa
import warnings
warnings.filterwarnings('ignore')

def load_audio_safe(path, target_length=16000):
    try:
        # Try loading with soundfile
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
            
        # Resample if needed
        if len(audio) != target_length:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_length)
            
        # Normalize and ensure length
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
            
        return audio
    except Exception as e:
        print(f"Warning: Could not load {path}, using silent fallback. Error: {str(e)}")
        return np.zeros(target_length)

class AudioDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir):
        self.clean_files = sorted([
            os.path.join(clean_dir, f) for f in os.listdir(clean_dir) 
            if f.endswith('.wav')
        ])
        self.noisy_files = sorted([
            os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir)
            if f.endswith('.wav')
        ])
        
        self.validate_files()
        
    def validate_files(self):
        if len(self.clean_files) != len(self.noisy_files):
            raise ValueError("Mismatched number of clean/noisy files")
            
        for clean, noisy in zip(self.clean_files, self.noisy_files):
            if os.path.basename(clean) != os.path.basename(noisy):
                raise ValueError(f"Filename mismatch: {clean} vs {noisy}")
            if not os.path.exists(clean):
                raise FileNotFoundError(f"Missing file: {clean}")
            if not os.path.exists(noisy):
                raise FileNotFoundError(f"Missing file: {noisy}")
    
    def __len__(self):
        return len(self.clean_files)
        
    def __getitem__(self, idx):
        clean = load_audio_safe(self.clean_files[idx])
        noisy = load_audio_safe(self.noisy_files[idx])
        
        return (
            torch.FloatTensor(noisy),
            torch.FloatTensor(clean)
        )

def get_dataloaders(config):
    data_config = config['data_config']
    
    train_set = AudioDataset(
        os.path.join(data_config['base_dir'], 'train/clean'),
        os.path.join(data_config['base_dir'], 'train/noisy')
    )
    
    val_set = AudioDataset(
        os.path.join(data_config['base_dir'], 'test/clean'),
        os.path.join(data_config['base_dir'], 'test/noisy')
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config['training_config']['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config['training_config']['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader