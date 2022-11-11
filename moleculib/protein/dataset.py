import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pathlib import Path
from typing import List, Union
from torch.utils.data import Dataset
from tempfile import gettempdir
from .datum import ProteinDatum
from .transform import ProteinTransform
from .utils import config, pids_file_to_list
from .alphabet import UNK_TOKEN
import traceback

from tqdm import tqdm

MAX_COMPLEX_SIZE = 32
PDB_METADATA_FIELDS = [('pid', str), ('num_res', int), ('standard', bool), ('resolution', float)]
PDB_METADATA_FIELDS += [(f'num_res_{idx}', int) for idx in range(MAX_COMPLEX_SIZE)]


class PDBMetadata:
    """
    Abstracts a pandas Dataframe into metadata for PDB
    Defines how the raw Datum is incorporated as a row in metadata
    """

    def __init__(self):
        series = {c: Series(dtype=t) for (c, t) in PDB_METADATA_FIELDS}
        self.df = DataFrame(series)

    def __len__(self):
        return len(self.df)

    def fetch(self,):
        raise NotImplementedError('')

    def write(self, filepath):
        # TODO(): this saving throws the following error
        # PerformanceWarning: your performance may suffer as PyTables will pickle object types that it cannot
        # map directly to c-types [inferred_type->mixed,key->block2_values] 
        # ------> [items->Index(['pid', 'standard', 'resolution'], dtype='object')]
        self.df.to_hdf(filepath, key='PDBMetadata')

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def add(self, datum: ProteinDatum):
        metrics = dict()
        metrics['pid'] = datum.pid
        metrics['standard'] = not (datum.residue_token == UNK_TOKEN).all()
        metrics['resolution'] = datum.pid
        metrics['num_res'] = datum.atom_coord.shape[0]

        # Note(Allan): eventually to be made into ComplexDatum.iter_chains() to
        # generate tensor views instantiated as ProteinDatum
        for chain_idx in range(np.max(datum.chain_token)):
            metrics[f'num_res_{chain_idx}'] = datum.atom_coord[datum.chain_token == chain_idx].shape[0]
        self.df = pd.concat((self.df, Series(metrics).to_frame().T))



class ProteinDataset(Dataset):
    """
    Holds ProteinDatum dataset with specified PDB IDs

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

    def __init__(self,
                 base_path: str,
                 metadata: PDBMetadata,
                 format: str = 'npz',
                 transform: ProteinTransform = None,
                 attrs: Union[List[str], str] = 'all'):

        super().__init__()
        self.base_path = Path(base_path)
        self.metadata = metadata
        self.format = format
        self.transform = transform

        # specific protein attributes
        protein_attrs = [
            'pid',
            'resolution',
            'residue_token',
            'residue_token',
            'residue_mask',
            'chain_token',
            'atom_token',
            'atom_coord',
            'atom_mask',
        ]

        if attrs == 'all':
            self.attrs = protein_attrs
        else:
            for attr in attrs:
                if attr not in protein_attrs:
                    raise AttributeError(f"attribute {attr} is invalid")
            self.attrs = attrs

    @classmethod
    def build(self, pdb_ids: List[str] = None, base_path: str = None, format='npz'):
        """
        Builds dataset from scratch given specified pdb_ids, prepares
        data and metadata for later use.
        """
        if pdb_ids is None:
            root = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
            pdb_ids = pids_file_to_list(root + '/data/pids_all.txt')
        if base_path is None:
            base_path = gettempdir()

        metadata = PDBMetadata()
        for pdb_id in tqdm(pdb_ids):
            try:
                datum = ProteinDatum.fetch_from_pdb(pdb_id, save_path=base_path, format=format)
            except KeyboardInterrupt: exit()
            except (ValueError, IndexError) as error:
                print(traceback.format_exc())
                print(error)
                continue
            except:
                breakpoint()
            if len(datum.sequence) != 0:
                metadata.add(datum)
        metadata.write(base_path + 'metadata.h5')
        return ProteinDataset(base_path, metadata=metadata, format=format)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        pdb_id = self.metadata[idx]['pid']
        filepath = os.path.join(self.base_path, f"{pdb_id}.{self.format}")
        protein = ProteinDatum.from_filepath(filepath, self.format)
        if self.transform is not None:
            protein = self.transform.transform(protein)
        return protein
