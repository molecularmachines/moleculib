
from ..abstract.dataset import PreProcessedDataset
from typing import List, Callable
import os
import pickle
from tqdm import tqdm


class ChromaDataset(PreProcessedDataset):

    def __init__(
            self, 
            base_path, 
            transform: List[Callable] = None, 
            monomeric: bool = False,
            shuffle=True,
        ):
        base_path = os.path.join(base_path, 'CHROMA_TRAIN.pyd')
        with open(base_path, 'rb') as fin:
            print('Loading data...')
            splits = pickle.load(fin)

        if monomeric:
            for split, list_ in splits.items():
                monomers = []
                for assembly in tqdm(list_): 
                    monomers.extend(assembly.protein_data)
                splits[split] = monomers
        
        super().__init__(splits, transform, shuffle, pre_transform=False)
