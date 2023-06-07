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
from .datum import ProteinDatum
from .transform import ProteinTransform
from .utils import pids_file_to_list
from tqdm import tqdm

MAX_COMPLEX_SIZE = 32
PDB_HEADER_FIELDS = [
    ("idcode", str),
    ("num_res", int),
    ("standard", bool),
    ("resolution", float),
]
CHAIN_COUNTER_FIELDS = [(f"num_res_{idx}", int) for idx in range(MAX_COMPLEX_SIZE)]
PDB_METADATA_FIELDS = PDB_HEADER_FIELDS + CHAIN_COUNTER_FIELDS

SAMPLE_PDBS = ["1C5E", "1C9O", "1CKU", "1CSE", "7ZKR", "7ZYS", "8AJQ", "8AQL", "8DCH"]


class PDBDataset(Dataset):
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
        protein_attrs = [
            "idcode",
            "resolution",
            "residue_token",
            "residue_index",
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

    def _is_in_filter(self, sample):
        return int(sample["id"]) in self.shard_indices

    def __len__(self):
        return len(self.metadata)

    def load_index(self, idx):
        header = self.metadata.iloc[idx]
        pdb_id = header["idcode"]
        filepath = os.path.join(self.base_path, f"{pdb_id}.mmtf")
        molecules = ProteinDatum.from_filepath(filepath)
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
    def _extract_datum_row(datum):
        is_standard = not (datum.residue_token == UNK_TOKEN).all()
        metrics = dict(
            idcode=datum.idcode,
            standard=is_standard,
            resolution=datum.resolution,
            num_res=len(datum.sequence),
        )
        for chain in range(np.max(datum.chain_token) + 1):
            num_residues = (datum.chain_token == chain).sum()
            metrics[f"num_res_{chain}"] = num_residues
        return Series(metrics).to_frame().T

    @staticmethod
    def _maybe_fetch_and_extract(pdb_id, save_path):
        try:
            datum = ProteinDatum.fetch_pdb_id(pdb_id, save_path=save_path)
        except KeyboardInterrupt:
            exit()
        except (ValueError, IndexError) as error:
            print(traceback.format_exc())
            print(error)
            return None
        except biotite.database.RequestError as request_error:
            print(request_error)
            return None
        if len(datum.sequence) == 0:
            return None
        return (datum, PDBDataset._extract_datum_row(datum))

    @classmethod
    def build(
        cls,
        pdb_ids: List[str] = None,
        save: bool = True,
        save_path: str = None,
        max_workers: int = 1,
        **kwargs,
    ):
        """
        Builds dataset from scratch given specified pdb_ids, prepares
        data and metadata for later use.
        """
        print(f"Extracting {len(pdb_ids)} PDB IDs with {max_workers} workers...")
        if pdb_ids is None:
            root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
            pdb_ids = pids_file_to_list(root + "/data/pids_all.txt")
        if save_path is None:
            save_path = mkdtemp()

        series = {c: Series(dtype=t) for (c, t) in PDB_METADATA_FIELDS}
        metadata = DataFrame(series)

        extractor = partial(cls._maybe_fetch_and_extract, save_path=save_path)
        if max_workers > 1:
            extraction = process_map(
                extractor, pdb_ids, max_workers=max_workers, chunksize=50
            )
        else:
            extraction = list(map(extractor, pdb_ids))

        extraction = filter(lambda x: x, extraction)
        data, metadata_ = list(map(list, zip(*extraction)))
        metadata = pd.concat((metadata, *metadata_), axis=0)

        if save:
            with open(str(Path(save_path) / "metadata.pyd"), "wb") as file:
                pickle.dump(metadata, file)

        return cls(base_path=save_path, metadata=metadata, **kwargs)


class MonomerDataset(PDBDataset):
    def __init__(
        self,
        base_path: str,
        pdb_ids: List[str] = None,
        metadata: pd.DataFrame = None,
        **kwargs,
    ):
        if base_path is None:
            base_path = mkdtemp()
            if pdb_ids is None:
                raise ValueError("pdb_ids must be specified if base_path is None")
            MonomerDataset.build(
                pdb_ids=pdb_ids, save_path=base_path, save=True, **kwargs
            )

        # read from base path if metadata is not built
        if metadata is None:
            with open(str(Path(base_path) / "metadata.pyd"), "rb") as file:
                metadata = pickle.load(file)
        metadata = metadata.reset_index()

        # flatten metadata with regards to num_res
        filtered = metadata.loc[metadata.index.repeat(MAX_COMPLEX_SIZE)]
        filtered["source"] = filtered.index
        filtered = filtered.reset_index()
        filtered["chain_indexes"] = pd.Series(np.zeros((len(filtered)), dtype=np.int32))
        for counter in range(MAX_COMPLEX_SIZE):
            filtered.loc[counter::MAX_COMPLEX_SIZE, "num_res"] = filtered.iloc[
                counter::MAX_COMPLEX_SIZE
            ][f"num_res_{counter}"]
            filtered.loc[counter::MAX_COMPLEX_SIZE, "chain_indexes"] = counter
        metadata = filtered[
            [col for (col, _) in PDB_HEADER_FIELDS] + ["chain_indexes", "source"]
        ]
        metadata = metadata[metadata["num_res"] > 0].reset_index()

        # initialize PDBDataset
        super().__init__(base_path=base_path, metadata=metadata, **kwargs)

    def parse(self, header, datum):
        chain_filter = header.chain_indexes == datum.chain_token
        values = list(vars(datum).values())
        proxy = values[0]

        if chain_filter.sum() == len(proxy):
            return datum

        chain_indexes = np.nonzero(chain_filter.astype(np.int32))[0]
        slice_min, slice_max = chain_indexes.min(), chain_indexes.max()

        def _cut_chain(obj):
            if type(obj) != np.ndarray and type(obj) != list:
                return obj
            return obj[slice_min:slice_max]

        values = list(map(_cut_chain, values))

        return ProteinDatum(*values)
