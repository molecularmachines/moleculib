from functools import reduce
from typing import Callable, List
import numpy as np

from tqdm import tqdm

class _TransformWrapper:
    
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        return reduce(lambda x, t: t.transform(x), self.transform, self.ds[idx])


class Dataset:

    def __init__(self, transform : List[Callable] = None):
        self.transform = transform

    def _getindex(self, index):
        raise NotImplementedError('Subclasses must implement this method')

    def __getitem__(self, idx):
        data = self._getitem(idx)
        if self.transform is not None:
            return reduce(lambda x, t: t.transform(x), self.transform, data)
        return data


class PreProcessedDataset:

    def __init__(self, splits, transform: List[Callable] = None, shuffle=True, pre_transform=False):
        self.splits = splits

        if shuffle:
            for split, data in list(self.splits.items()):
                print(f'Shuffling {split}...')
                np.random.shuffle(data)
                self.splits[split] = data
                
        self.transform = transform
        if pre_transform:
            if self.transform is None:
                raise ValueError('Cannot pre-transform without a transform')
            for split, data in list(self.splits.items()):
                self.splits[split] = [ reduce(lambda x, t: t.transform(x), self.transform, datum) for datum in tqdm(data) ]
        else:
            if self.transform is not None:
                for split, data in list(self.splits.items()):
                    self.splits[split] = _TransformWrapper(data, self.transform)

    def __len__(self):
        return sum([len(data) for data in self.splits.values()])
