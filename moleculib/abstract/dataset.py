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


class PreProcessedDataset:

    def __init__(self, splits, transform: List[Callable] = None, shuffle=True, pre_transform=False):
        self.splits = splits

        if shuffle:
            for split, data in list(self.splits.items()):
                print(f'Shuffling {split}...')
                self.splits[split] = np.random.permutation(data)
                
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
