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

from .alphabet import UNK_TOKEN
from .datum import NucleicDatum, dna_res_tokens, rna_res_tokens
from .utils import pids_file_to_list
from tqdm import tqdm

#NOTE: TBD:
# MAX_COMPLEX_SIZE = 32
PDB_HEADER_FIELDS = [
    ("idcode", str),
    ("num_res", int),
    ("num_rna_chains",int),
    ("num_dna_chains",int),
    ("standard", bool),
    ("resolution", float),
]
CHAIN_COUNTER_FIELDS = [(f"num_res_{idx}", int) for idx in range(MAX_COMPLEX_SIZE)]
PDB_METADATA_FIELDS = PDB_HEADER_FIELDS + CHAIN_COUNTER_FIELDS

SAMPLE_PDBS = ["1C5E", "1C9O", "1CKU", "1CSE", "7ZKR", "7ZYS", "8AJQ", "8AQL", "8DCH"]



class PDBDataset(Dataset):
    """
    Holds NucleicDatum dataset with specified PDB IDs

    Arguments:
    ----------
    base_path : str
        directory to store all PDB files
    pdb_ids : List[str]
        list of all protein IDs that should be in the dataset
    format : str
        the file format for each PDB file, either "npz" or "pdb"
    attrs : Union[str, List[str]]
        a partial list of protein attributes that should be in each protein
    """

    def __init__(
        self,
        base_path: str,
        transform: ProteinTransform = None,
        attrs: Union[List[str], str] = "all",
        metadata: pd.DataFrame = None,
        max_resolution: float = None,
        min_sequence_length: int = None,
        max_sequence_length: int = None,
        frac: float = 1.0,
        preload: bool = False,
        preload_num_workers: int = 10,
    ):
        super().__init__()
        self.base_path = Path(base_path)
        if metadata is None:
            with open(str(self.base_path / "metadata.pyd"), "rb") as file:
                metadata = pickle.load(file)
        self.metadata = metadata
        self.transform = transform

        if max_resolution is not None:
            self.metadata = self.metadata[self.metadata["resolution"] <= max_resolution]

        if min_sequence_length is not None:
            self.metadata = self.metadata[
                self.metadata["num_res"] >= min_sequence_length
            ]

        if max_sequence_length is not None:
            self.metadata = self.metadata[
                self.metadata["num_res"] <= max_sequence_length
            ]

        # shuffle and sample
        self.metadata = self.metadata.sample(frac=frac).reset_index(drop=True)

        # specific protein attributes
        nuc_attrs = [
            "idcode",
            "resolution",
            "sequence", #added
            "nuc_token", #mod
            "nuc_index",#mod
            "nuc_mask",#mod
            "chain_token",
            "atom_token",
            "atom_coord",
            "atom_mask",
        ]

        if attrs == "all":
            self.attrs = nuc_attrs
        else:
            for attr in attrs:
                if attr not in nuc_attrs:
                    raise AttributeError(f"attribute {attr} is invalid")
            self.attrs = attrs

    def _is_in_filter(self, sample):
        return int(sample["id"]) in self.shard_indices

    def __len__(self):
        return len(self.metadata)

    def load_index(self, idx):
        header = self.metadata.iloc[idx]
        pdb_id = header["idcode"]
        filepath = os.path.join(self.base_path, f"{pdb_id}.mmtf")
        molecules = NucleicDatum.from_filepath(filepath)
        return self.parse(header, molecules)

    def parse(self, molecules):
        raise NotImplementedError("PDBDataset is an abstract class")

    def __getitem__(self, idx):
        molecule = self.data[idx] if hasattr(self, "data") else self.load_index(idx)
        if self.transform is not None:
            for transformation in self.transform:
                molecule = transformation.transform(molecule)
        return molecule

    @staticmethod
    def _extract_statistics(datum):
        """
        Gets the following statistics for the 
        df row for each datum:
            idcode#
            num_res#
            num_rna_chains #
            num_dna_chains #
             
            standard#
            resolution#
        """
        is_standard = not (datum.nuc_token == UNK_TOKEN).all()
        metrics = dict(
            idcode=datum.idcode,
            standard=is_standard,
            resolution=datum.resolution,
            num_res=len(datum.sequence),
        )

        def check_strictly_increasing(lst):
            for i in range(1, len(lst)):
                if lst[i] < lst[i-1]:
                    return False
            return True

        num_rna_chains=0
        num_dna_chains=0

        for chain in range(np.max(datum.chain_token) + 1):
            #getting chain length for each chain
            chain_residues = (datum.chain_token == chain) #bool list, true where the chain is
            num_residues = chain_residues.sum()
            metrics[f"num_res_{chain}"] = num_residues

            #NOTE: this check is prob not relevant, tbd if needed
            if check_strictly_increasing(datum.chain_token) == False:
                raise Exception("The datum chain tokens are not strictly increasing")
            #getting chain type
            chain_indices = [i for i, val in enumerate(chain_residues) if val]
            random_index = random.choice(chain_indices)
            random_nuc_token = datum.nuc_token[random_index]
            if random_nuc_token in rna_res_tokens:
                num_rna_chains+=1
            elif random_nuc_token in dna_res_tokens:
                num_dna_chains+=1
            else:
                raise Exception("The datum nuc token didn't fit RNA or DNA tokens")
        return Series(metrics).to_frame().T