from moleculib.abstract.dataset import PreProcessedDataset
from moleculib.nucleic.datum import NucleicDatum
from typing import List, Callable
from functools import reduce
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import numpy as np
import pickle


        
class RNADataset(Dataset):
    def __init__(self, datums, transform=None):
        self.datums = datums
        self.transform = transform if transform is not None else []
        self.splits = { 'train': self }

    def __len__(self):
        return len(self.datums)

    def __getitem__(self, idx):
        datum = self.datums[idx]
        if self.transform:
            datum = reduce(lambda x, t: t.transform(x), self.transform, datum)
        return datum
    
    
def split_datum_by_chain(datum):
    """ Split a datum into multiple datums based on chain tokens. """
    indices = np.where(np.diff(datum.chain_token) != 0)[0] + 1
    # print(indices)
    split_points = np.split(np.arange(len(datum.chain_token)), indices)
    # print(split_points)
    # print("datum.nuc_token[split_points[0]]:   ", datum.nuc_token[split_points[0]])
    
    new_datums = []
    for idx_array in split_points:
        first_idx = idx_array[0]
        last_idx = idx_array[-1]
        new_datums.append(
            NucleicDatum(
                idcode=datum.idcode,
                resolution=datum.resolution,
                sequence=datum.sequence[first_idx:last_idx + 1], 
                nuc_token=datum.nuc_token[idx_array],
                nuc_index=datum.nuc_index[idx_array],
                nuc_mask=datum.nuc_mask[idx_array],
                chain_token=datum.chain_token[idx_array],
                atom_token=datum.atom_token[idx_array],
                atom_coord=datum.atom_coord[idx_array],
                atom_mask=datum.atom_mask[idx_array]
            )
        )
    return new_datums  

