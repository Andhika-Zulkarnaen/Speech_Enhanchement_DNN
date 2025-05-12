# audio_enhancer_module.py
import torch.nn as nn

class AudioEnhancer(nn.Module):
    def __init__(self, input_size=256, hidden_size=512, output_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size)
        )
    
    def forward(self, x):
        return self.net(x)