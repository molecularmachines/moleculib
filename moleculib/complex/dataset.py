
from ..protein.dataset import PreProcessedDataset
from typing import List, Callable
import os
import pickle


class ChromaDataset(PreProcessedDataset):

    def __init__(self, base_path, transform: List[Callable] = None, shuffle=True):
        base_path = os.path.join(base_path, 'CHROMA_TRAIN.pyd')
        with open(base_path, 'rb') as fin:
            print('Loading data...')
            splits = pickle.load(fin)
        super().__init__(splits, transform, shuffle, pre_transform=False)
