from moleculib.abstract.dataset import PreProcessedDataset
from moleculib.nucleic.datum import NucleicDatum
from typing import List, Callable
from functools import reduce
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import numpy as np
import RNA
import pickle



        
class RNADataset(PreProcessedDataset):
    def __init__(self, datums, transform: List[Callable] = None, shuffle = True, train_split=0.9):
        num_train = int(len(datums) * train_split)
        num_val = len(datums) - num_train
        
        if num_val == 0 :
            print("Error, number of validation datums is 0")
        
        if shuffle:
            np.random.shuffle(datums)
        
        train_data = datums[:num_train]
        val_data = datums[num_train:]
        
        #get casp data:
        path_casp = "/mas/projects/molecularmachines/db/PREPROCESSED/test_CASP_len_8.pkl"
        #unpickle the data:
        with open(path_casp, 'rb') as f:
            casp_data = pickle.load(f)
            
        #get RNA Puzzles data:
        path_rnapuzzles = "/mas/projects/molecularmachines/db/PREPROCESSED/test_RNA-Puzzles_len_21.pkl"
        with open(path_rnapuzzles, 'rb') as f:
            rnapuzzles_data = pickle.load(f)

        #get rna art data:
        path_rna_art = "/mas/projects/molecularmachines/db/PREPROCESSED/test_RNART1_len_27.pkl"
        with open(path_rna_art, 'rb') as f:
            rna_art_data = pickle.load(f)


        splits = {
            'train': train_data,
            'val': val_data,
            'casp': casp_data,
            'puzzles': rnapuzzles_data, 
            'rnart': rna_art_data 
        }

        super().__init__(splits, transform=transform, shuffle=shuffle)


    # def __len__(self):
    #     return len(self.datums)

    def __getitem__(self, idx):
        datum = self.datums[idx]
        if self.transform:
            datum = reduce(lambda x, t: t.transform(x), self.transform, datum)
        return datum
    

def secondary_dot_bracket_to_contact_map(dot_bracket):
    length = len(dot_bracket)
    contact_map = np.zeros((length, length), dtype=int)
    stack = []

    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            j = stack.pop()
            contact_map[i, j] = 1
            contact_map[j, i] = 1
    
    return contact_map
        
def split_datum_by_chain(datum):
    """ Split a datum into multiple datums based on chain tokens. """
    indices = np.where(np.diff(datum.chain_token) != 0)[0] + 1
    # print(indices)
    split_points = np.split(np.arange(len(datum.chain_token)), indices)
    # print(split_points)
    # print("datum.nuc_token[split_points[0]]:   ", datum.nuc_token[split_points[0]])
    
    token_to_rna_letter = {'0': 'A',
                            '1': 'U',
                            '2': 'T',
                            '3': 'G',
                            '4': 'C',
                            '5': 'I',
                            '13': 'N',
                            '6': 'A', #DNA
                            '11': 'U',#DNA
                            '10': 'T',#DNA
                            '8': 'G',#DNA
                            '7': 'C',#DNA
                            '9': 'I',#DNA
                            '12': 'N'} #PAD
    
    new_datums = []
    for idx_array in split_points:
        first_idx = idx_array[0]
        last_idx = idx_array[-1]
        
        residue_token = datum.nuc_token[idx_array]
        
        seq =''
        for r in residue_token:
            seq += token_to_rna_letter[str(r)]
        fc = RNA.fold_compound(seq)
        mfe_structure, mfe = fc.mfe() #Example of mfe structure "....(...((.())))"
        contact_pairs = secondary_dot_bracket_to_contact_map(mfe_structure)
        
        
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
                atom_mask=datum.atom_mask[idx_array],
                contact_map = contact_pairs,
                # id = datum.id
            )
        )
    return new_datums  

