import json
import sys
import torch
from torch.utils.data import Dataset

import os
import pickle
import traceback
from functools import partial
from pathlib import Path
from tempfile import mkdtemp
from typing import List, Union

import biotite
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map

from moleculib.protein.alphabet import UNK_TOKEN
from moleculib.protein.transform import ProteinTransform
from moleculib.sequence.datum import SeqDatum
import torch.nn.functional as F

from tqdm import tqdm

def onehot_encode_datum(sequencedatum:SeqDatum):
    residue_tokens = torch.Tensor(sequencedatum.residue_token).to(torch.int64)
    one_hot_tensor = F.one_hot(residue_tokens, num_classes=23)
    one_hot_tensor = one_hot_tensor.unsqueeze(0)
    return one_hot_tensor.float()

class ElutedLigandDataset(Dataset):
    def __init__(
            self,
            base_path: str,
            transform: ProteinTransform = None,
            encode_one_hot:bool=True
    ):
        super().__init__()
        self.base_path = Path(base_path)
        self.transform = transform
        self.encode_one_hot = encode_one_hot
        self.data_frame = pd.read_csv(base_path)
        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)
        self.splits = dict(train=self)
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        peptide = SeqDatum.from_sequence("peptide_{}".format(idx),row['peptide'])
        mhc = SeqDatum.from_sequence("mhc_{}".format(idx),row['pseudo_sequence'])
        label = row['label']
        
        if self.transform is not None:
            for transformation in self.transform:
                peptide = transformation.transform(peptide)
        if self.encode_one_hot:
            encoderd_peptide = onehot_encode_datum(peptide).squeeze(0)
        
        return (encoderd_peptide,torch.Tensor([label]).long())


class MHCBindingAffinityDataset(Dataset):
    def __init__(
            self,
            base_path: str,
            transform: ProteinTransform = None
    ):
        super().__init__()
        self.base_path = Path(base_path)
        self.transform = transform
        self.data_frame = pd.read_csv(base_path)
        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)
        self.splits = dict(train=self)
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        peptide = SeqDatum.from_sequence("peptide_{}".format(idx),row['peptide'])
        mhc = SeqDatum.from_sequence("mhc_{}".format(idx),row['pseudosequence'])
        label = row['ba']
        
        if self.transform is not None:
            for transformation in self.transform:
                peptide = transformation.transform(peptide)
               
        return (peptide,mhc,label)
    
class GFPFitnessDataset(Dataset):
    def __init__(
            self,
            base_path: str,
            transform: ProteinTransform = None,
            encode_one_hot:bool=True,
            encode_lm:bool=False,
            lm_model:str="esm"
    ):
        super().__init__()
        self.base_path = Path(base_path)
        self.transform = transform
        self.data_frame = pd.read_csv(base_path)
        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)
        self.splits = dict(train=self)
        self.encode_one_hot = encode_one_hot
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        
        row = self.data_frame.iloc[idx]
        peptide = SeqDatum.from_sequence("peptide_{}".format(idx),row['sequence'])
        label = row['target']
        
        if self.transform is not None:
            for transformation in self.transform:
                peptide = transformation.transform(peptide)
        
        encoded_tensor = torch.Tensor(peptide.residue_token).long()
        output_label = torch.Tensor([label]).float()
        if self.encode_one_hot:
            encoderd_tensor = onehot_encode_datum(SeqDatum.from_tokens(encoded_tensor)).squeeze(0)
        
        return (encoderd_tensor,output_label)
