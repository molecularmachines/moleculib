from moleculib.abstract.dataset import PreProcessedDataset
from typing import List, Callable
import os
import pickle


class RNADataset(PreProcessedDataset):

    def __init__(self, base_path, transform: List[Callable] = None, shuffle=True):
        path = os.path.join(base_path, 'RNA_data_list.pkl')
        with open(path, 'rb') as fin:
            print(f'Loading data from {path}')
            splits = {'train': pickle.load(fin)}
        super().__init__(splits, transform, shuffle)