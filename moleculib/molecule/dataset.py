import os
import pickle
import traceback
from functools import partial
from pathlib import Path
from tempfile import gettempdir
from typing import List, Union
import numpy as np
import biotite
import pandas as pd
from pandas import Series
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from .datum import PDBMoleculeDatum, QM9Datum
from .transform import (
    MoleculeTransform,
    MoleculePad,
    DescribeGraph,
    Permuter,
    Centralize,
    AtomFeatures,
    NormalizeProperties
)
from .utils import pids_file_to_list
from .alphabet import elements


class PDBMoleculeDataset(Dataset):
    """
    Holds MoleculeDatum dataset with specified PDB IDs

    Arguments:
    ----------
    base_path : str
        directory to store all PDB files
    pdb_ids : List[str]
        list of all Molecule IDs that should be in the dataset
    format : str
        the file format for each PDB file, either "npz" or "pdb"
    attrs : Union[str, List[str]]
        a partial list of Molecule attributes that should be in each Molecule
    """

    def __init__(
        self,
        base_path: str,
        transform: List[MoleculeTransform] = None,
        attrs: Union[List[str], str] = "all",
        metadata: pd.DataFrame = None,
        max_resolution: float = None,
        min_atom_count: int = None,
        max_atom_count: int = None,
        frac: float = 1.0,
        preload: bool = False,
        preload_num_workers: int = 10,
    ):
        super().__init__()
        self.base_path = Path(base_path)
        if metadata is None:
            with open(str(self.base_path / "metadata_mol.pyd"), "rb") as file:
                metadata = pickle.load(file)
        self.metadata = metadata
        self.transform = transform

        if max_resolution is not None:
            self.metadata = self.metadata[self.metadata["resolution"] <= max_resolution]

        if min_atom_count is not None:
            self.metadata = self.metadata[self.metadata["atom_count"] >= min_atom_count]

        if max_atom_count is not None:
            self.metadata = self.metadata[self.metadata["atom_count"] <= max_atom_count]

        # shuffle and sample
        if frac < 1.0:
            self.metadata = self.metadata.sample(frac=frac).reset_index(drop=True)
        print(f"Loaded metadata with {len(self.metadata)} samples")

        self.splits = {"train": self.metadata}  # TODO: patch to kheiron

        # specific Molecule attributes
        molecule_attrs = [
            "idcode",
            "resolution",
            "chain_id",
            "res_id",
            "res_name",
            "atom_token",
            "atom_coord",
            "atom_name",
            "b_factor",
            "molecule_mask",
        ]

        if attrs == "all":
            self.attrs = molecule_attrs
        else:
            for attr in attrs:
                if attr not in molecule_attrs:
                    raise AttributeError(f"attribute {attr} is invalid")
            self.attrs = attrs

        self.preload = preload
        if self.preload:
            molecules = []
            for idx in range(len(self.metadata.index)):
                molecules.append(self.load_index(idx))
            self.molecules = molecules

    def load_index(self, idx):
        pdb_id = self.metadata.iloc[idx]["idcode"]
        molecule_idx = self.metadata.iloc[idx]["molecule_idx"]
        filepath = os.path.join(self.base_path, f"{pdb_id}.mmtf")
        molecule = PDBMoleculeDatum.from_filepath(filepath, molecule_idx=molecule_idx)
        return molecule

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        molecule = self.molecules[idx] if self.preload else self.load_index(idx)
        if self.transform is not None:
            for transformation in self.transform:
                molecule = transformation.transform(molecule)
        return molecule

    @staticmethod
    def _extract_datum_row(datum):
        metrics = [
            dict(
                idcode=datum.idcode,
                resolution=datum.resolution,
                chain_id=datum.chain_id[mask][0],
                res_id=datum.res_id[mask][0],
                res_name=datum.res_name[mask][0],
                atom_count=len(datum.atom_token[mask]),
                bond_count=datum.bonds[mask].get_bond_count(),
                molecule_idx=i,
            )
            for i, mask in enumerate(datum.atom_mask)
        ]
        return pd.concat(list(map(Series, metrics)), axis=1).T

    @staticmethod
    def _maybe_fetch_and_extract(pdb_id, save_path):
        try:
            datum = PDBMoleculeDatum.fetch_pdb_id(pdb_id, save_path=save_path)
        except KeyboardInterrupt:
            exit()
        except (ValueError, IndexError, biotite.InvalidFileError) as error:
            print(traceback.format_exc())
            print(error)
            return None
        except KeyError as e:
            print(e)
            print(pdb_id)
            exit()
        except biotite.database.RequestError as request_error:
            print(request_error)
            return None
        if len(datum.atom_token) == 0:
            return None
        if len(datum.atom_mask) == 0:
            return None
        return PDBMoleculeDataset._extract_datum_row(datum)

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
        if pdb_ids is None:
            root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
            pdb_ids = pids_file_to_list(root + "/data/pids_all.txt")
        if save_path is None:
            save_path = gettempdir()

        extractor = partial(cls._maybe_fetch_and_extract, save_path=save_path)
        if max_workers > 1:
            rows = process_map(
                extractor, pdb_ids, max_workers=max_workers, chunksize=50
            )
        else:
            rows = list(map(extractor, pdb_ids))
        rows = filter(lambda row: row is not None, rows)

        metadata = pd.concat(rows, axis=0).reset_index(drop=True)
        if save:
            with open(str(Path(save_path) / "metadata_mol.pyd"), "wb") as file:
                pickle.dump(metadata, file)

        return cls(base_path=save_path, metadata=metadata, **kwargs)


class QM9Dataset(Dataset):
    def __init__(
        self,
        base_path="QM9",
        molecule_transform: List = [],
        permute=False,
        centralize=True,
        use_atom_features=True,
    ):
        with open(os.path.join(base_path, "data.pyd"), "rb") as f:
            print("Loading data...")
            self.data = pickle.load(f)
        self.graph = DescribeGraph()
        self.padding = MoleculePad(29)
        self.permute = Permuter() if permute else None
        self.centralize = Centralize() if centralize else None
        self.atom_features = AtomFeatures()
        self.normalize_properties = NormalizeProperties()
        self.use_atom_features = use_atom_features
        self.splits = {"train": self}  # FIXME: patch to kheiron

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        idcode = datum["name"]
        atom_coord = datum["pos"]
        atom_token = datum["z"]
        atom_mask = np.ones_like(atom_token, dtype=bool)
        properties = np.squeeze(datum["y"])

        bonds = datum["edge_index"].T
        # allan come back here
        bonds = np.concatenate([bonds, np.ones((len(bonds), 1))], axis=1).astype(
            np.int32
        )

        datum = QM9Datum(
            idcode,
            atom_token,
            atom_coord,
            atom_mask,
            bonds,
            properties=properties,
            stds=np.ones_like(properties),
        )
        datum = self.normalize_properties.transform(datum)
        
        if self.centralize:
            datum = self.centralize.transform(datum)

        if self.permute is not None:
            datum = self.permute(datum)

        datum = self.graph.transform(datum)
        datum = self.padding.transform(datum)
        if self.use_atom_features:
            datum = self.atom_features.transform(datum)
        return datum


from functools import reduce
import os
import pickle
from typing import Callable, List

from tqdm.contrib.concurrent import process_map


class _TransformWrapper:
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return reduce(lambda x, t: t.transform(x), self.transform, self.ds[idx])


class PreProcessedDataset:
    def __init__(
        self,
        splits,
        transform: List[Callable] = None,
        shuffle=True,
        pre_transform=False,
    ):
        self.splits = splits

        if shuffle:
            for split, data in list(self.splits.items()):
                print(f"Shuffling {split}...")
                self.splits[split] = np.random.permutation(data)

        self.transform = transform
        if pre_transform:
            if self.transform is None:
                raise ValueError("Cannot pre-transform without a transform")
            for split, data in list(self.splits.items()):
                self.splits[split] = [
                    reduce(lambda x, t: t.transform(x), self.transform, datum)
                    for datum in tqdm(data)
                ]
        else:
            if self.transform is not None:
                for split, data in list(self.splits.items()):
                    self.splits[split] = _TransformWrapper(data, self.transform)


class QM9Processed(PreProcessedDataset):
    def __init__(self, base_path, transform: List[Callable] = None, shuffle=True):
        base_path = os.path.join(base_path, "data.pyd")
        with open(base_path, "rb") as fin:
            print("Loading data...")
            splits = pickle.load(fin)
        super().__init__(splits, transform, shuffle, pre_transform=False)
