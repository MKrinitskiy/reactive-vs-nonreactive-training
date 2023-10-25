import torch
import torch.nn as nn



class CNNmodel(torch.nn.Module):
    def __init__(self):
        
        super(CNNmodel, self).__init__()
        self.layers = torch.nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(), # 68 x 90
            nn.Linear(210304, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid())
    
    def forward(self, image_batch):
        logits = self.layers(image_batch)
        return logits
    
