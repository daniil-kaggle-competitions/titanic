import torch.nn as nn
import torch.nn.functional as F

class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return F.sigmoid(self.stack(x))