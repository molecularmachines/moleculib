import os
from pathlib import Path
from typing import List, Union
from torch.utils.data import Dataset

from .datum import ProteinDatum
from .transform import ProteinTransform
from .utils import config


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
                 pdb_ids: List[str],
                 format: str = 'npz',
                 transform: ProteinTransform = None,
                 attrs: Union[List[str], str] = 'all'):

        super().__init__()
        self.base_path = Path(base_path)
        self.pdb_ids = pdb_ids
        self.format = format
        self.transform = transform

        # specific protein attributes
        protein_attrs = [
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
    def fetch_from_pdb(self, pdb_ids, base_path=None, format='npz'):
        if base_path is None:
            base_path = config['cache_dir']
        for pdb_id in pdb_ids:
            pid_path = os.path.join(base_path, f"{pdb_id}.{format}")
            if os.path.exists(pid_path):
                continue
            ProteinDatum.fetch_from_pdb(pdb_id, save_path=base_path, format=format)
        return ProteinDataset(base_path, pdb_ids, format)

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        pdb_id = self.pdb_ids[idx]
        filepath = os.path.join(self.base_path, f"{pdb_id}.{self.format}")
        protein = ProteinDatum.from_filepath(filepath, self.format)
        if self.transform is not None:
            protein = self.transform.transform(protein)
        return protein
