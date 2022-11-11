import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pathlib import Path
from functools import partial
from typing import List, Union
from torch.utils.data import Dataset
from tempfile import gettempdir
from .datum import ProteinDatum
from .transform import ProteinTransform
from .utils import config, pids_file_to_list
from .alphabet import UNK_TOKEN
import traceback

from tqdm.contrib.concurrent import process_map

MAX_COMPLEX_SIZE = 32
PDB_METADATA_FIELDS = [
    ("pid", str),
    ("num_res", int),
    ("standard", bool),
    ("resolution", float),
]
PDB_METADATA_FIELDS += [(f"num_res_{idx}", int) for idx in range(MAX_COMPLEX_SIZE)]


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

    def __init__(
        self,
        base_path: str,
        metadata: DataFrame,
        format: str = "npz",
        transform: ProteinTransform = None,
        attrs: Union[List[str], str] = "all",
    ):

        super().__init__()
        self.base_path = Path(base_path)
        self.metadata = metadata
        self.format = format
        self.transform = transform

        # specific protein attributes
        protein_attrs = [
            "pid",
            "resolution",
            "residue_token",
            "residue_token",
            "residue_mask",
            "chain_token",
            "atom_token",
            "atom_coord",
            "atom_mask",
        ]

        if attrs == "all":
            self.attrs = protein_attrs
        else:
            for attr in attrs:
                if attr not in protein_attrs:
                    raise AttributeError(f"attribute {attr} is invalid")
            self.attrs = attrs

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        pdb_id = self.metadata.iloc[idx]["pid"]
        filepath = os.path.join(self.base_path, f"{pdb_id}.{self.format}")
        protein = ProteinDatum.from_filepath(filepath, self.format)
        if self.transform is not None:
            protein = self.transform.transform(protein)
        return protein

    @staticmethod
    def _extract_datum_row(datum):
        is_standard = not (datum.residue_token == UNK_TOKEN).all()
        metrics = dict(
            pid=datum.pid,
            standard=is_standard,
            resolution=datum.pid,
            num_res=len(datum.sequence),
        )
        for chain in range(np.max(datum.chain_token)):
            num_residues = (datum.chain_token == chain).sum()
            metrics[f"num_res_{chain}"] = num_residues
        return Series(metrics).to_frame().T

    @staticmethod
    def _fetch_and_extract(pdb_id, save_path):
        try:
            datum = ProteinDatum.fetch_from_pdb(
                pdb_id, save_path=save_path, format=format
            )
        except KeyboardInterrupt:
            exit()
        except (ValueError, IndexError) as error:
            print(traceback.format_exc())
            print(error)
            return None
        if len(datum.sequence) == 0:
            return None
        return ProteinDataset._extract_datum_row(datum)

    @classmethod
    def build(cls, pdb_ids: List[str] = None, save_path: str = None, max_workers: int = 1, format="npz"):
        """
        Builds dataset from scratch given specified pdb_ids, prepares
        data and metadata for later use.
        """
        if pdb_ids is None:
            root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
            pdb_ids = pids_file_to_list(root + "/data/pids_all.txt")
        if save_path is None:
            save_path = gettempdir()

        series = {c: Series(dtype=t) for (c, t) in PDB_METADATA_FIELDS}
        metadata = DataFrame(series)
        extractor = partial(cls._fetch_and_extract, save_path=save_path)
        rows = process_map(extractor, pdb_ids, max_workers=max_workers)
        rows = filter(lambda row: row is not None, rows)
        metadata = pd.concat((metadata, *rows), axis=0)

        return ProteinDataset(save_path, metadata=metadata, format=format)
