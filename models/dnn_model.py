import torch.nn as nn

class DNNSpeechEnhancer(nn.Module):
    def __init__(self, input_size=16000, hidden_sizes=[512, 512, 512], output_size=16000, dropout=0.2):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
            
        layers.append(nn.Linear(prev_size, output_size))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # Input shape: [batch_size, sequence_length]
        if x.dim() == 2:
            return self.net(x)
        elif x.dim() == 1:
            return self.net(x.unsqueeze(0))
        else:
            raise ValueError(f"Invalid input dimension: {x.dim()}")