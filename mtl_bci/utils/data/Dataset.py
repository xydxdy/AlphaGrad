from torch.utils.data import Dataset as BaseDataset
import torch
import numpy as np
import copy

class Dataset(BaseDataset):
    """Custom dataset

    Args:
        BaseDataset (torch.utils.data.Dataset)
    """
    def __init__(self, data, labels, transform=None, unique_inverse=True, expand_dims_axis=None, to_tensor=True, transpose_axes=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.expand_dims_axis = expand_dims_axis
        self.to_tensor = to_tensor
        self.transpose_axes = transpose_axes
        
        if unique_inverse:
            u, indices = np.unique(labels, return_inverse=True)
            self.labels = indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Load and provide the data and label"""
        data = copy.deepcopy(self.data[idx])
        labels = copy.deepcopy(self.labels[idx])
        if self.transform:
            data = self.transform(data)
        if self.expand_dims_axis is not None:
            data = np.expand_dims(data, axis=self.expand_dims_axis)
        if self.transpose_axes:
            data = np.transpose(data, axes=self.transpose_axes)
        if self.to_tensor:
            data = torch.from_numpy(data).float()
        return {"data": data, "labels": labels}
