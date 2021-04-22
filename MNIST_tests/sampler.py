import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class IIDBatchSampler:
    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):
            indices = np.where(torch.rand(self.length) < (self.minibatch_size / self.length))[0]
            if indices.size > 0:
                yield indices

    def __len__(self):
        return self.iterations

def get_loader(minibatch_size, iterations):
    def minibatch_loader(dataset):
            return DataLoader(
                dataset,
                batch_sampler=IIDBatchSampler(dataset, minibatch_size, iterations),
                num_workers=4
            )
    return minibatch_loader