import json
import sys
import torch
from embed.seq_embeddings import ProteinSeqEmbeddings
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
from moleculib.sequence.datum import SeqDatum
from moleculib.protein.transform import ProteinTransform

from tqdm import tqdm


class ElutedLigandDataset(Dataset):
    def __init__(
            self,
            base_path: str,
            transform: ProteinTransform = None,
            min_sequence_length: int = 1,
            max_sequence_length: int = 14
    ):
        super().__init__()
        self.base_path = Path(base_path)
        self.transform = transform
        self.data_frame = pd.read_csv(base_path)
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        peptide = row['peptide']
        mhc = row['pseudo_sequence']
        label = row['label']
        
        if self.transform is not None:
            for transformation in self.transform:
                peptide = transformation.transform(peptide)
                
        return (peptide,mhc,label)

