import os
import pickle
import traceback
from functools import partial
from pathlib import Path
from tempfile import gettempdir
from typing import List, Union
from scipy.sparse.csgraph import laplacian
import numpy as np
import biotite
import pandas as pd
from pandas import Series
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from .datum import PDBMoleculeDatum, QM9Datum, MoleculeDatum, RSDatum
from .transform import (
    MoleculeTransform,
    MoleculePad,
    DescribeGraph,
    Permuter,
    Centralize,
    AtomFeatures,
    StandardizeProperties
)
from .utils import pids_file_to_list, extract_rdkit_mol_properties


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
        full_data=True,
        num_train=None,
        num_val=None,
        num_test=None,
        molecule_transform: List = [],
        permute=False,
        centralize=True,
        use_atom_features=True,
        standardize=True,
        shuffle=False,
        _split="train",
        _data=None,
    ):
        if _data is None:
            data_path = "full_data.pyd" if full_data else "data.pyd"
            with open(os.path.join(base_path, data_path), "rb") as f:
                _data = pickle.load(f)
        if num_train is None or not full_data:
            num_val = len(_data) // 10
            num_train = len(_data) - 2 * num_val
        num_val = num_val if num_val is not None else len(_data) - num_train - num_test
        num_test = (
            num_test if num_test is not None else len(_data) - num_train - num_val
        )
        if shuffle:
            rng = np.random.default_rng(seed=shuffle)
            rng.shuffle(_data, axis=0)
        self.data = _data
        if _split == "train":
            self.data = self.data[:num_train]
        elif _split == "valid":
            self.data = self.data[num_train : num_train + num_val]
        elif _split == "test":
            self.data = self.data[num_train + num_val : num_train + num_val + num_test]
        print(f"Loaded {len(self.data)} {_split} datapoints!")
        self.graph = DescribeGraph()
        self.padding = MoleculePad(29)
        self.permute = Permuter() if permute else None
        self.centralize = Centralize() if centralize else None
        self.atom_features = AtomFeatures()
        self.standardize = standardize
        if standardize:
            self.standardize = StandardizeProperties()
        self.use_atom_features = use_atom_features
        if _split == "train":
            self.splits = {
                "train": self,
                "valid": self.__class__(
                    base_path,
                    full_data,
                    num_train,
                    num_val,
                    num_test,
                    molecule_transform,
                    permute,
                    centralize,
                    use_atom_features,
                    shuffle=False,
                    _split="valid",
                    _data=_data,
                ),
                "test": self.__class__(
                    base_path,
                    full_data,
                    num_train,
                    num_val,
                    num_test,
                    molecule_transform,
                    permute,
                    centralize,
                    use_atom_features,
                    shuffle=False,
                    _split="test",
                    _data=_data,
                ),
            }  # FIXME: patch to kheiron

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
            idcode=idcode,
            atom_token=atom_token,
            atom_coord=atom_coord,
            atom_mask=atom_mask,
            bonds=bonds,
            properties=properties,
            stds=np.ones_like(properties),
        )
        if self.standardize:
            datum = self.standardize.transform(datum)

        if self.centralize:
            datum = self.centralize.transform(datum)

        if self.permute is not None:
            datum = self.permute(datum)

        datum = self.graph.transform(datum)
        datum = self.padding.transform(datum)
        if self.use_atom_features:
            datum = self.atom_features.transform(datum)
        return datum


class RSDataset(Dataset):
    def __init__(
        self,
        base_path="ChIRo",
        molecule_transform: List = [],
        permute=False,
        use_atom_features=True,
        max_num_atoms=50,
        _split="train",
    ):
        print(f"Loading {_split} data...")
        if _split == "train":
            f = "train_RS_classification_enantiomers_MOL_326865_55084_27542.pkl"
        elif _split == "valid":
            f = "validation_RS_classification_enantiomers_MOL_70099_11748_5874.pkl"
        elif _split == "test":
            f = "test_RS_classification_enantiomers_MOL_69719_11680_5840.pkl"
        else:
            raise ValueError("Invalid split")

        self.data = pd.read_pickle(os.path.join(base_path, f))
        self.max_num_atoms = max_num_atoms
        if self.max_num_atoms is not None:
            self.data = self.data[
                self.data.rdkit_mol_cistrans_stereo.apply(
                    lambda x: x.GetNumAtoms() <= self.max_num_atoms
                )
            ]
        self.data = self.data.iloc
        print(f"Loaded {len(self)} datapoints!")
        self.padding = MoleculePad(max_num_atoms)
        self.permute = Permuter() if permute else None
        self.atom_features = AtomFeatures()
        self.use_atom_features = use_atom_features
        if _split == "train":
            self.splits = {
                "train": self,
                "valid": self.__class__(
                    base_path,
                    molecule_transform,
                    permute,
                    use_atom_features,
                    max_num_atoms,
                    "valid",
                ),
                "test": self.__class__(
                    base_path,
                    molecule_transform,
                    permute,
                    use_atom_features,
                    max_num_atoms,
                    "test",
                ),
            }  # FIXME: patch to kheiron

    def __len__(self):
        return self.data[:].shape[0]

    def __getitem__(self, idx):
        """
        Some lines taken from
        https://github.com/keiradams/ChIRo/blob/main/model/datasets_samplers.py#L167
        """
        datum = self.data[idx]
        atom_token, atom_coord, bonds, adj = extract_rdkit_mol_properties(
            datum.rdkit_mol_cistrans_stereo
        )
        atom_mask = np.ones_like(atom_token, dtype=bool)
        properties = np.zeros((2,))
        properties[datum.RS_label_binary] = 1.0

        L = laplacian(adj, normed=False).astype(np.float32)

        datum = RSDatum(
            atom_token=atom_token,
            atom_coord=atom_coord,
            atom_mask=atom_mask,
            bonds=bonds,
            properties=properties,
            adjacency=adj,
            laplacian=L,
        )

        if self.permute is not None:
            datum = self.permute(datum)

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
