from enum import Enum

import numpy as np
import torch
from torch.autograd import Variable


class BatchState(Enum):
    CPU_APPENDING = 1
    CPU_STORING = 2
    CUDA_STORING = 3


class Batch:
    def __init__(self):
        self.images = []
        self.masks = None
        self.targets = []

        self.state = BatchState.CPU_APPENDING

    def __len__(self):
        return len(self.images)

    def append(self,
               image: np.ndarray,
               target: float):
        if self.state is not BatchState.CPU_APPENDING:
            raise ValueError(f'You can append to batch only on CPU_APPENDING state. '
                             f'But there was an attempt on {self.state} state')

        self.images.append(image)
        self.targets.append(target)

    def to_tensor(self):
        if self.state is not BatchState.CPU_APPENDING:
            raise ValueError(f'You can convert to tensor only from CPU_APPENDING state. '
                             f'But there was an attempt on {self.state} state')

        self.images = torch.stack(tuple(torch.tensor((i / 255.)[np.newaxis, :, :]) for i in self.images))
        self.targets = torch.reshape(torch.tensor(self.targets), (-1, 1))
        self.state = BatchState.CPU_STORING

    def to_cuda(self, cuda_device, to_variable: bool):
        if self.state is not BatchState.CPU_STORING:
            raise ValueError(f'You can load to cuda device only on CPU_STORING state. '
                             f'But there was an attempt on {self.state} state')

        self.images = self.images.float()
        self.targets = self.targets.float()

        if to_variable:
            self.images = Variable(self.images)
            self.targets = Variable(self.targets)

        self.images = self.images.to(cuda_device)
        self.targets = self.targets.to(cuda_device)

        self.state = BatchState.CUDA_STORING
