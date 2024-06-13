import os
import pickle
import traceback
from functools import partial
from pathlib import Path
from tempfile import gettempdir
from typing import List, Union, Callable
from scipy.sparse.csgraph import laplacian
import numpy as np
import biotite
import pandas as pd
from pandas import Series
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from .transform import (
    MoleculeTransform,
    MoleculePad,
    DescribeGraph,
    Permuter,
    Centralize,
    AtomFeatures,
    StandardizeProperties,
    SortAtoms,
    PairPad,
)
from .utils import pids_file_to_list, extract_rdkit_mol_properties
from .alphabet import elements
from rdkit import Chem
from rdkit.Chem import AllChem
import random
from rdkit import RDLogger
from functools import reduce
import lmdb
from moleculib.molecule.datum import (
    CrossdockDatum,
    PDBBindDatum,
    PDBMoleculeDatum,
    QM9Datum,
    RSDatum,
    ReactDatum,
    MISATODatum,
    DensityDatum,
)
import biotite.structure.io.pdb as pdb
import biotite.structure.io as strucio
import mrcfile
from sklearn.cluster import KMeans
import json
from ase.calculators.vasp import VaspChargeDensity
import lz4.frame
import tempfile
import h5py
from moleculib.molecule.h5_to_pdb import create_pdb
import multiprocessing
import threading
import logging

# Suppress RDKit prints
RDLogger.DisableLog("rdApp.*")


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
        _max_atoms=29,
        _padding=True,
        to_split=True,
    ):
        base_path = "/mas/projects/molecularmachines/db/QM9"
        if _data is None:
            data_path = "full_data.pyd" if full_data else "data.pyd"
            with open(os.path.join(base_path, data_path), "rb") as f:
                _data = pickle.load(f)
        if num_train is None or not full_data:
            num_val = len(_data) // 10
            num_train = len(_data) - 2 * num_val
        if to_split:
            num_val = (
                num_val if num_val is not None else len(_data) - num_train - num_test
            )
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
        self.padding = MoleculePad(_max_atoms) if _padding else False
        self.permute = Permuter() if permute else None
        self.centralize = Centralize() if centralize else None
        self.sorter = SortAtoms()
        self.atom_features = AtomFeatures()
        self.standardize = standardize
        if standardize:
            self.standardize = StandardizeProperties()
        self.use_atom_features = use_atom_features
        if _split == "train" and to_split:
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

        # datum = self.sorter.transform(datum)

        if self.permute is not None:
            datum = self.permute.transform(datum)

        datum = self.graph.transform(datum)

        if self.padding:
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


# class QM9Processed(PreProcessedDataset):
#     def __init__(self, base_path, transform: List[Callable] = None, shuffle=True):
#         base_path = os.path.join(base_path, "data.pyd")
#         with open(base_path, "rb") as fin:
#             print("Loading data...")
#             splits = pickle.load(fin)
#         super().__init__(splits, transform, shuffle, pre_transform=False)


# class CrossdockDataset(Dataset):

#     def __init__(self, max_ligand_atoms=29, max_protein_atoms=400, _split="train"):
#         super().__init__()

#         base_path = "/mas/projects/molecularmachines/db/crossdocked_targetdiff/"
#         self.processed_path = os.path.join(
#             base_path, "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
#         )
#         self.split_file = os.path.join(base_path, "crossdocked_pocket10_pose_split.pkl")
#         self.index_file = os.path.join(base_path, "crossdocked_pocket10_pose_index.pkl")
#         self.db = None

#         self.padding = PairPad()
#         self.max_ligand_atoms = max_ligand_atoms
#         self.max_protein_atoms = max_protein_atoms

#         with open(self.index_file, "rb") as f:
#             data = pickle.load(f)
#         ligand_atoms = data["ligand_atoms"]
#         ligand_keys = set()
#         protein_atoms = data["protein_atoms"]
#         protein_keys = set()
#         for key, value in ligand_atoms:
#             if value <= self.max_ligand_atoms or self.max_ligand_atoms == -1:
#                 ligand_keys.add(key)
#             else:
#                 break
#         for key, value in protein_atoms:
#             if value <= self.max_protein_atoms or self.max_protein_atoms == -1:
#                 protein_keys.add(key)
#             else:
#                 break
#         with open(self.split_file, "rb") as f:
#             split_keys = set(pickle.load(f)[_split])
#         self.keys = ligand_keys & protein_keys & split_keys
#         self.keys = list(self.keys)
#         print(f"For {_split} loaded {len(self.keys)} pairs")
#         if _split == "train":
#             self.splits = {
#                 "train": self,
#                 "val": self.__class__(
#                     self.max_ligand_atoms, self.max_protein_atoms, _split="val"
#                 ),
#                 "test": self.__class__(
#                     self.max_ligand_atoms, self.max_protein_atoms, _split="test"
#                 ),
#             }
#             self.splits = {k: v for k, v in self.splits.items() if len(v) > 0}

#     def _connect_db(self):
#         """
#         Establish read-only database connection
#         """
#         assert self.db is None, "A connection has already been opened."
#         self.db = lmdb.open(
#             self.processed_path,
#             map_size=10 * (1024 * 1024 * 1024),  # 10GB
#             create=False,
#             subdir=False,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )

#     def _close_db(self):
#         self.db.close()
#         self.db = None
#         self.keys = None

#     def __len__(self):
#         return len(self.keys)

#     def __getitem__(self, idx):
#         data, key = self.get_ori_data(idx)
#         filename = data["ligand_filename"]
#         atom_token = np.array(data["ligand_element"])
#         atom_coord = np.array(data["ligand_pos"])
#         bonds = data["ligand_bond_index"]
#         bonds = np.row_stack((bonds, data["ligand_bond_type"])).T
#         atom_features = np.array(data["ligand_atom_feature"])

#         protein_token = np.array(data["protein_element"])
#         protein_coord = np.array(data["protein_pos"])

#         datum = CrossdockDatum(
#             key=key,
#             filename=filename,
#             atom_token=atom_token,
#             atom_coord=atom_coord,
#             atom_mask=np.ones_like(atom_token),
#             bonds=bonds,
#             atom_features=atom_features,
#             protein_token=protein_token,
#             protein_coord=protein_coord,
#             protein_mask=np.ones_like(protein_token),
#         )

#         datum = self.padding.transform(
#             datum,
#             {
#                 "atom_token": self.max_ligand_atoms,
#                 "atom_coord": self.max_ligand_atoms,
#                 "atom_mask": self.max_ligand_atoms,
#                 "bonds": self.max_ligand_atoms,
#                 "atom_features": self.max_ligand_atoms,
#                 "protein_token": self.max_protein_atoms,
#                 "protein_coord": self.max_protein_atoms,
#                 "protein_mask": self.max_protein_atoms,
#             },
#         )
#         return datum

#     def get_ori_data(self, idx):
#         if self.db is None:
#             self._connect_db()
#         key = self.keys[idx]
#         data = pickle.loads(self.db.begin().get(eval(f"b'{key}'")))
#         return data, key


# class PDBBindDataset(Dataset):
#     def __init__(
#         self, max_ligand_atoms=29, max_protein_atoms=400, _split="train", refined=True
#     ):
#         super().__init__()
#         self.max_ligand_atoms = max_ligand_atoms
#         self.max_protein_atoms = max_protein_atoms
#         if refined:
#             self.base_path = "/mas/projects/molecularmachines/db/PDBBind/refined-set"
#             self.index_path = os.path.join(
#                 self.base_path, "index/INDEX_refined_data.2020"
#             )
#         else:
#             self.base_path = (
#                 "/mas/projects/molecularmachines/db/PDBbind_v2020-other-PL/"
#             )
#             self.index_path = os.path.join(
#                 self.base_path, "index/INDEX_general_PL_data.2020"
#             )

#         self.index = self._load_index(_split)
#         print(f"Loaded {self.index.shape[0]} {_split} datapoints")

#         self.padding = PairPad()
#         self.elements = elements.assign(
#             symbol=lambda df: df.symbol.str.lower()
#         ).symbol.tolist()  # TODO:

#         if _split == "train":
#             self.splits = {
#                 "train": self,
#                 # "val": self.__class__(
#                 #     self.max_ligand_atoms, self.max_protein_atoms, _split="val"
#                 # ),
#                 "test": self.__class__(
#                     self.max_ligand_atoms, self.max_protein_atoms, _split="test"
#                 ),
#             }
#             self.splits = {k: v for k, v in self.splits.items() if len(v) > 0}

#     def _load_index(self, split):
#         KMAP = {"Ki": 1, "Kd": 2, "IC50": 3}

#         all_files = os.listdir(self.base_path)
#         all_index = []
#         with open(self.index_path, "r") as f:
#             lines = f.readlines()
#         for line in lines:
#             if line.startswith("#"):
#                 continue
#             index, res, year, pka, kv = line.split("//")[0].strip().split()

#             kind = [v for k, v in KMAP.items() if k in kv]
#             assert len(kind) == 1
#             if index in all_files:
#                 all_index.append([index, res, year, pka, kind[0]])

#         all_index = np.array(all_index)
#         with open(os.path.join(self.base_path, "index/lengths.pkl"), "rb") as f:
#             atoms_count = pickle.load(f)
#             prot_len = np.array(atoms_count["prot_len"]).squeeze()
#             lig_len = np.array(atoms_count["lig_len"]).squeeze()
#         protm = prot_len <= self.max_protein_atoms
#         ligm = lig_len <= self.max_ligand_atoms
#         sub_index = all_index[protm & ligm]

#         with open(f"{self.base_path}/timesplit_test", "r") as f:
#             test_split = [l.strip("\n") for l in f.readlines()]
#         split_index = []
#         for d in sub_index:
#             if split == "train" and d[0] in test_split:
#                 continue
#             if split == "test" and d[0] not in test_split:
#                 continue
#             split_index.append(d)

#         return np.array(split_index)

#     def __len__(self):
#         return self.index.shape[0]

#     def get_pdb_id(self, pdb_id):
#         idx = np.where(self.index[:, 0] == pdb_id)[0][0]
#         return self.__getitem__(idx)

#     def __getitem__(self, idx):
#         pdb_id, _, _, pka, _ = self.index[idx]
#         pka = float(pka)
#         protein_coord, protein_token = self._get_protein_pocket(pdb_id)
#         atom_coord, atom_token, bonds, charge = self._get_ligand(pdb_id)

#         datum = PDBBindDatum(
#             pdb_id=pdb_id,
#             pka=np.array(pka),
#             atom_token=atom_token,
#             atom_coord=atom_coord,
#             atom_mask=np.ones_like(atom_token),
#             charge=charge,
#             bonds=bonds,
#             protein_coord=protein_coord,
#             protein_token=protein_token,
#             protein_mask=np.ones_like(protein_token),
#         )

#         datum = self.padding.transform(
#             datum,
#             {
#                 "atom_token": self.max_ligand_atoms,
#                 "atom_coord": self.max_ligand_atoms,
#                 "atom_mask": self.max_ligand_atoms,
#                 "charge": self.max_ligand_atoms,
#                 "bonds": self.max_ligand_atoms,
#                 "protein_token": self.max_protein_atoms,
#                 "protein_coord": self.max_protein_atoms,
#                 "protein_mask": self.max_protein_atoms,
#             },
#         )
#         return datum

#     def _get_protein_pocket(self, pdb_id):
#         filepath = os.path.join(self.base_path, pdb_id, f"{pdb_id}_pocket.pdb")
#         pdb_file = pdb.PDBFile.read(filepath)
#         atom_array = pdb.get_structure(pdb_file, model=1)
#         coord = atom_array.coord
#         token = [self.elements.index(e.lower()) + 1 for e in atom_array.element]

#         return np.array(coord), np.array(token)

#     def _get_ligand(self, pdb_id):
#         filepath = os.path.join(self.base_path, pdb_id, f"{pdb_id}_ligand.sdf")
#         mol = strucio.load_structure(filepath)
#         token = [self.elements.index(e.lower()) + 1 for e in mol.element]
#         return (
#             mol.coord,
#             np.array(token),
#             mol.bonds._bonds.astype(np.int32),
#             mol.charge.astype(np.int32),
#         )


def _decompress_file(filepath):
    with lz4.frame.open(filepath, mode="rb") as fp:
        filecontent = fp.read()
    return filecontent


def _read_vasp(filecontent):
    # Write to tmp file and read using ASE
    tmpfd, tmppath = tempfile.mkstemp(prefix="tmpdeepdft")
    tmpfile = os.fdopen(tmpfd, "wb")
    tmpfile.write(filecontent)
    tmpfile.close()
    vasp_charge = VaspChargeDensity(filename=tmppath)
    os.remove(tmppath)
    density = vasp_charge.chg[-1]  # separate density
    atoms = vasp_charge.atoms[-1]  # separate atom positions
    return density, atoms, np.zeros(3)  # TODO: Can we always assume origin at 0,0,0?


import torch.nn.functional as F
from e3nn import o3


def rotate_voxel(shape, cell, density, rotated_grid):
    """
    Rotate the volumetric data using trilinear interpolation.
    :param shape: voxel shape, tensor of shape (3,)
    :param cell: cell vectors, tensor of shape (3, 3)
    :param density: original density, tensor of shape (n_grid,)
    :param rotated_grid: rotated grid coordinates, tensor of shape (n_grid, 3)
    :return: rotated density, tensor of shape (n_grid,)
    """
    density = density.view(1, 1, *shape)
    rotated_grid = rotated_grid.view(1, *shape, 3)
    shape = torch.FloatTensor(shape)
    grid_cell = cell / shape.view(3, 1)
    normalized_grid = (2 * rotated_grid @ torch.linalg.inv(grid_cell) - shape + 1) / (
        shape - 1
    )
    return F.grid_sample(
        density, torch.flip(normalized_grid, [-1]), mode="bilinear", align_corners=False
    )


def _calculate_grid_pos(density, origin, cell):
    # Calculate grid positions
    ngridpts = np.array(density.shape)  # grid matrix
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / density.shape[0],
        np.arange(ngridpts[1]) / density.shape[1],
        np.arange(ngridpts[2]) / density.shape[2],
        indexing="ij",
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, cell)
    grid_pos = grid_pos + origin
    return grid_pos


# def rotating_pool_worker(dataset, rng, queue):
#     while True:
#         for index in rng.permutation(len(dataset)).tolist():
#             queue.put(dataset[index])


# def transfer_thread(queue: multiprocessing.Queue, datalist: list):
#     while True:
#         for index in range(len(datalist)):
#             datalist[index] = queue.get()


# class RotatingPoolData(Dataset):
#     """
#     Wrapper for a dataset that continously loads data into a smaller pool.
#     The data loading is performed in a separate process and is assumed to be IO bound.
#     """

#     def __init__(self, dataset, pool_size, **kwargs):
#         super().__init__(**kwargs)
#         self.pool_size = pool_size
#         self.parent_data = dataset
#         self.rng = np.random.default_rng()
#         logging.debug("Filling rotating data pool of size %d" % pool_size)
#         self.data_pool = [
#             self.parent_data[i]
#             for i in self.rng.integers(
#                 0, high=len(self.parent_data), size=self.pool_size, endpoint=False
#             ).tolist()
#         ]
#         self.loader_queue = multiprocessing.Queue(2)

#         # Start loaders
#         self.loader_process = multiprocessing.Process(
#             target=rotating_pool_worker,
#             args=(self.parent_data, self.rng, self.loader_queue),
#         )
#         self.transfer_thread = threading.Thread(
#             target=transfer_thread, args=(self.loader_queue, self.data_pool)
#         )
#         self.loader_process.start()
#         self.transfer_thread.start()

#     def __len__(self):
#         return self.pool_size

#     def __getitem__(self, index):
#         return self.data_pool[index]


import torch


class DensityDataDir(Dataset):
    def __init__(
        self,
        max_atoms=29,
        grid_size=36,
        samples=1000,
        _split="train",
        _rotated=True,
        to_split=True,
        rotate_voxel=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.directory = "/mas/projects/molecularmachines/db/qm9_vasp"
        self.padding = PairPad()
        self.max_atoms = max_atoms
        self.grid_size = grid_size
        self.samples = samples
        self.rotate_voxel = rotate_voxel
        if to_split:
            split = json.load(open(os.path.join(self.directory, "split.json")))[_split]
        else:
            split = []
        self.member_list = []
        for s in range(134):
            self.member_list.extend(
                sorted(
                    [
                        f
                        for f in os.listdir(os.path.join(self.directory, str(s)))
                        if f.endswith(".lz4")
                        and ((int(f.split(".")[0]) in split) or not to_split)
                    ]
                )
            )

#         print(f"Loaded {_split} {len(self)} datapoints")
#         if _split == "train":
#             self.splits = {"train": RotatingPoolData(self, 300) if _rotated else self}
#             test = self.__class__(
#                 max_atoms=max_atoms,
#                 grid_size=grid_size,
#                 samples=5000,
#                 _split="test",
#             )
#             self.splits["test"] = RotatingPoolData(test, 90) if _rotated else test
#             valid = self.__class__(
#                 max_atoms=max_atoms,
#                 grid_size=grid_size,
#                 samples=5000,
#                 _split="validation",
#             )
#             self.splits["valid"] = RotatingPoolData(valid, 30) if _rotated else valid

#     def __len__(self):
#         return len(self.member_list)

#     def extractfile(self, index):
#         filename = self.member_list[index]
#         path = os.path.join(
#             self.directory, f"{int(filename.split('.')[0]) // 1000}", filename
#         )
#         if filename.endswith(".npz"):
#             with np.load(path) as f:
#                 return {
#                     "density": f["density"],
#                     "coord": f["coord"],
#                     "token": f["token"],
#                     "grid": f["grid"],
#                     "filename": filename,
#                 }

        filecontent = _decompress_file(path)
        density, atoms, origin = _read_vasp(filecontent)
        cell = atoms.get_cell()
        grid_pos = _calculate_grid_pos(density, origin, cell)

        return {
            "density": density,
            "coord": atoms.positions,
            "token": atoms.numbers,
            "grid": grid_pos,
            "filename": filename,
            "cell": cell,
        }

#     def _process(self, index):
#         dp = self.extractfile(index)
#         file_num = dp["filename"].split(".")[0]
#         path = os.path.join(
#             self.directory, f"{int(file_num) // 1000}", file_num + ".npz"
#         )
#         np.savez(
#             path,
#             density=dp["density"],
#             grid=dp["grid"],
#             coord=dp["coord"],
#             token=dp["token"],
#         )

#     def __getitem__(self, index):
#         dp = self.extractfile(index)

        density = dp["density"]
        grid = dp["grid"]
        cell = dp["cell"]
        coord = dp["coord"]
        # print(density.shape)
        if self.rotate_voxel:
            rot = o3.rand_matrix()
            center = torch.Tensor(cell).sum(dim=0) / 2
            rotated_coord = (torch.Tensor(coord) - center) @ rot.t() + center
            rotated_grid = (torch.Tensor(grid) - center) @ rot + center
            rotated_density = rotate_voxel(
                density.shape, torch.Tensor(cell), torch.Tensor(density), rotated_grid
            )

            # density = rotated_density.numpy().squeeze()
            grid = rotated_grid.numpy()
            coord = rotated_coord.numpy()

        # dx = np.diff(grid[:, :, :, 0], axis=0).mean() / 0.1
        # dy = np.diff(grid[:, :, :, 1], axis=1).mean() / 0.1
        # dz = np.diff(grid[:, :, :, 2], axis=2).mean() / 0.1
        # density = density * (dx * dy * dz)
        # print(density.shape)
        # print(grid.shape)
        # print(coord.shape)
        if self.grid_size:
            # density, grid = self.sample_neighborhood(density, grid)
            density, grid = self.sample(density, grid, self.samples)

        datum = DensityDatum(
            density=density,
            grid=grid,
            atom_coord=coord,
            atom_token=dp["token"],
            atom_mask=np.ones_like(dp["token"]),
            bonds=None,
        )

#         datum = self.padding.transform(
#             datum,
#             {
#                 "atom_token": self.max_atoms,
#                 "atom_coord": self.max_atoms,
#                 "atom_mask": self.max_atoms,
#             },
#         )

#         return datum

#     def com(self, density_array):
#         # Create arrays of indices along each axis
#         x_indices, y_indices, z_indices = np.indices(density_array.shape)

#         # Calculate the total mass
#         total_mass = np.sum(density_array)

#         # Calculate the center of mass along each axis
#         center_x = np.sum(x_indices * density_array) / total_mass
#         center_y = np.sum(y_indices * density_array) / total_mass
#         center_z = np.sum(z_indices * density_array) / total_mass

#         return np.round([center_x, center_y, center_z]).astype(np.int32)

#     def neighborhood(self, center, density_array, n):
#         # Calculate the center of mass indices
#         center_x, center_y, center_z = center

#         # Calculate the boundaries for indexing
#         x_min = max(0, center_x - n // 2)
#         x_max = min(density_array.shape[0], center_x + n // 2)
#         y_min = max(0, center_y - n // 2)
#         y_max = min(density_array.shape[1], center_y + n // 2)
#         z_min = max(0, center_z - n // 2)
#         z_max = min(density_array.shape[2], center_z + n // 2)

#         if (x_max - x_min) < n:
#             if x_max == density_array.shape[0]:
#                 x_min -= n - (x_max - x_min)
#             else:
#                 x_max += n - (x_max - x_min)
#         if (y_max - y_min) < n:
#             if y_max == density_array.shape[1]:
#                 y_min -= n - (y_max - y_min)
#             else:
#                 y_max += n - (y_max - y_min)
#         if (z_max - z_min) < n:
#             if z_max == density_array.shape[2]:
#                 z_min -= n - (z_max - z_min)
#             else:
#                 z_max += n - (z_max - z_min)

#         return x_min, x_max, y_min, y_max, z_min, z_max

#     def sample_neighborhood(self, density, grid):
#         assert (
#             density.shape[0] >= self.grid_size
#         ), f"{density.shape[0]} < {self.grid_size}"
#         assert (
#             density.shape[1] >= self.grid_size
#         ), f"{density.shape[1]} < {self.grid_size}"
#         assert (
#             density.shape[2] >= self.grid_size
#         ), f"{density.shape[2]} < {self.grid_size}"

#         center = self.com(density)
#         center += np.random.randint(-self.grid_size // 2, self.grid_size // 2, 3)
#         x_min, x_max, y_min, y_max, z_min, z_max = self.neighborhood(
#             center, density, self.grid_size
#         )
#         density = density[
#             x_min:x_max,
#             y_min:y_max,
#             z_min:z_max,
#         ].reshape(-1)
#         grid = grid[
#             x_min:x_max,
#             y_min:y_max,
#             z_min:z_max,
#         ].reshape((-1, 3))
#         return density, grid

#     def sample(self, density, grid_pos, num_probes):
#         # Sample probes on the calculated grid
#         probe_choice_max = np.prod(grid_pos.shape[0:3])
#         probe_choice = np.random.randint(probe_choice_max, size=num_probes)
#         probe_choice = np.unravel_index(probe_choice, grid_pos.shape[0:3])
#         probe_pos = grid_pos[probe_choice]
#         probe_target = density[probe_choice]
#         return probe_target, probe_pos


# class MISATO(Dataset):
#     def __init__(self, neighborhood=15.0, _split="train") -> None:
#         super().__init__()
#         self.base_path = "/mas/projects/molecularmachines/db/MISATO"
#         self.data = h5py.File(os.path.join(self.base_path, "MD.hdf5"))
#         self.h5_properties = [
#             "trajectory_coordinates",
#             "atoms_type",
#             "atoms_number",
#             "atoms_residue",
#             "atoms_element",
#             "molecules_begin_atom_index",
#             # "frames_rmsd_ligand",
#             # "frames_distance",
#             # "frames_interaction_energy",
#             # "frames_bSASA",
#         ]
#         self.neighborhood = neighborhood
#         self.index = (
#             open(os.path.join(self.base_path, f"{_split}_MD.txt"), "r")
#             .read()
#             .split("\n")
#         )

#         print(f"Loading {_split} {len(self)} datapoints")

#         if _split == "train":
#             self.splits = {"train": self}
#             self.splits["valid"] = self.__class__(
#                 _split="valid",
#             )
#             self.splits["test"] = self.__class__(
#                 _split="test",
#             )

#     def __len__(self):
#         return len(self.index)

#     def __getitem__(self, index):
#         pdb_id = self.index[index]
#         dp = self.get_entries(pdb_id)

#         traj_coord = dp["trajectory_coordinates"]
#         token = dp["atoms_number"]
#         mol_idx = dp["molecules_begin_atom_index"][-1]

#         atom_token = token[mol_idx:]
#         atom_coord = traj_coord[:, mol_idx:]
#         atom_mask = np.ones_like(atom_token)

#         protein_token = token[:mol_idx]
#         atoms_residue = dp["atoms_residue"][:mol_idx]
#         atoms_type = dp["atoms_type"][:mol_idx]
#         protein_coord = traj_coord[:, :mol_idx]
#         protein_mask = np.ones_like(protein_token)

#         datum = MISATODatum(
#             pdb_id=pdb_id,
#             atom_token=atom_token,
#             atom_coord=atom_coord,
#             atom_mask=atom_mask,
#             bonds=None,
#             protein_token=protein_token,
#             protein_coord=protein_coord,
#             protein_mask=protein_mask,
#             atoms_residue=atoms_residue,
#             atoms_type=atoms_type,
#         )
#         if self.neighborhood:
#             return datum.keep_neighborhood(self.neighborhood)

#         return datum

#     def get_entries(self, pdbid):
#         h5_entries = {}
#         for h5_property in self.h5_properties:
#             h5_entries[h5_property] = self.data.get(pdbid + "/" + h5_property)
#         return h5_entries

#     def pdb_str(self, index, frame):
#         pdb_id = self.index[index]
#         dp = self.get_entries(pdb_id)
#         return "\n".join(
#             create_pdb(
#                 dp["trajectory_coordinates"][frame],
#                 dp["atoms_type"],
#                 dp["atoms_number"],
#                 dp["atoms_residue"],
#                 dp["molecules_begin_atom_index"],
#             )
#         )


# class MISATODensity(Dataset):

#     def __init__(
#         self, max_atoms=50, samples=1000, _split="train", _rotated=True
#     ) -> None:
#         super().__init__()
#         self.base_path = "/mas/projects/molecularmachines/db/MISATO"
#         self.data = h5py.File(os.path.join(self.base_path, "QM.hdf5"))
#         self.samples = samples
#         self.max_atoms = max_atoms
#         self.padding = PairPad()

#         if _split == "train":
#             self.index = self._get_index("train")
#             self.index = np.concatenate([self.index, self._get_index("valid", s=50)])
#             self.index = np.concatenate([self.index, self._get_index("test", s=200)])
#         elif _split == "valid":
#             self.index = self._get_index("valid", e=50)
#         elif _split == "test":
#             self.index = self._get_index("test", e=200)

#         print(f"Loading {_split} {len(self)} datapoints")

#         if _split == "train":
#             self.splits = {"train": RotatingPoolData(self, 300) if _rotated else self}
#             test = self.__class__(
#                 max_atoms=max_atoms,
#                 samples=5000,
#                 _split="test",
#             )
#             self.splits["test"] = RotatingPoolData(test, 90) if _rotated else test
#             valid = self.__class__(
#                 max_atoms=max_atoms,
#                 samples=5000,
#                 _split="valid",
#             )
#             self.splits["valid"] = RotatingPoolData(valid, 30) if _rotated else valid

#     def _get_index(self, split, s=0, e=-1):
#         index = (
#             open(os.path.join(self.base_path, f"{split}_MD.txt"), "r")
#             .read()
#             .split("\n")[s:e]
#         )
#         sizes = np.load(os.path.join(self.base_path, f"{split}_sizes.npy"))[s:e]
#         index = np.array(index)[np.where((sizes <= self.max_atoms) & (sizes != -1))]
#         return index

#     def __len__(self):
#         return len(self.index)

#     def __getitem__(self, index):
#         pdb_id = self.index[index]
#         grid, density, coord, token = self.extract(pdb_id)
#         # print(pdb_id)
#         # sample grid and density
#         if self.samples:
#             probes = np.random.randint(0, len(grid), self.samples)
#             grid = grid[probes]
#             density = density[probes]

#         datum = DensityDatum(
#             density=density.astype(np.float32) / 63,
#             grid=grid.astype(np.float32),
#             atom_coord=coord.astype(np.float32),
#             atom_token=token.astype(np.int32),
#             atom_mask=np.ones_like(token),
#             bonds=None,
#         )

#         datum = self.padding.transform(
#             datum,
#             {
#                 "atom_token": self.max_atoms,
#                 "atom_coord": self.max_atoms,
#                 "atom_mask": self.max_atoms,
#             },
#         )

#         return datum

#     def extract(self, pdb_id):
#         element = np.array(
#             self.data[pdb_id.upper()]["atom_properties"]["atom_names"]
#         ).astype(np.int32)
#         coord = self.data[pdb_id.upper()]["atom_properties"]["atom_properties_values"][
#             :, [0, 1, 2]
#         ][:, [2, 1, 0]]
#         grid, density = self.read_mrc(
#             os.path.join(
#                 self.base_path, "densities_gfn2w_mrc", f"{pdb_id.lower()}.mrc"
#             ),
#             down=1,
#         )
#         grid, density, coord = self.align(grid, density, coord, element)

#         return grid, density, coord, element

#     def read_mrc(self, mrcfilename, down=1):
#         """
#         Read a mrc file and return the xyz and density values at the given level
#         if given
#         """
#         xyz = []
#         with mrcfile.open(mrcfilename) as emd:
#             nx, ny, nz = emd.header["nx"], emd.header["ny"], emd.header["nz"]
#             dx, dy, dz = emd.voxel_size["x"], emd.voxel_size["y"], emd.voxel_size["z"]
#             xyz = np.meshgrid(
#                 np.arange(0, nx * dx, dx),
#                 np.arange(0, ny * dy, dy),
#                 np.arange(0, nz * dz, dz),
#                 indexing="ij",
#             )
#             xyz = np.asarray(xyz)
#             density = emd.data.flatten("F").reshape(nx, ny, nz)
#             return xyz, density

#     def align(self, grid, density, coord, element):
#         # rough align
#         cc = grid[:, *density.nonzero()].T.mean(0)
#         ac = coord.mean(0)
#         coord = coord - ac + cc

#         hmask = element != 1  # filter hydrogens
#         hcoord = coord[np.where(hmask)]

#         grid = grid.reshape((3, -1)).T
#         density = density.flatten()

#         kmeans = KMeans(
#             n_clusters=hcoord.shape[0],
#             init=hcoord,
#             n_init=1,
#         )
#         kmeans.fit(grid[np.where(density > 130)])  # keep only high density points
#         centers = kmeans.cluster_centers_

#         # fine align
#         shift = centers - hcoord
#         deviation = np.log(((shift - shift.mean(0)) ** 2).sum(-1))
#         # filter out outliers
#         shift_mask = deviation < -4
#         if shift_mask.sum() == 0:
#             shift_mask = deviation < -3
#             if shift_mask.sum() == 0:
#                 shift_mask = deviation < -2
#                 if shift_mask.sum() == 0:
#                     return grid, density, coord

#         coord = coord + (shift * shift_mask[:, None]).sum(0) / shift_mask.sum()
#         return grid, density, coord


# class InterleavedDataset(Dataset):
#     def __init__(self, dataset1, dataset2):
#         self.dataset1 = dataset1
#         self.dataset2 = dataset2
#         self.total_length = len(dataset1) + len(dataset2)

#     def __len__(self):
#         return self.total_length

#     def __getitem__(self, index):
#         if index < len(self.dataset1):
#             return self.dataset1[index]
#         else:
#             return self.dataset2[index - len(self.dataset1)]


# class DensityDataset(InterleavedDataset):
#     def __init__(self, max_atoms=50, samples=1000, _rotated=True) -> None:
#         qm9_vasp = DensityDataDir(
#             max_atoms=max_atoms, samples=samples, _rotated=False, to_split=False
#         )
#         misato = MISATODensity(max_atoms=max_atoms, samples=samples, _rotated=False)
#         super().__init__(qm9_vasp, misato)
#         valid = misato.splits["valid"]
#         test = misato.splits["test"]
#         self.splits = {
#             "train": RotatingPoolData(self, 300) if _rotated else self,
#             "valid": RotatingPoolData(valid, 30) if _rotated else valid,
#             "test": RotatingPoolData(test, 90) if _rotated else test,
#         }


# class ReactDataset(Dataset):
#     def __init__(self, max_atoms=70):
#         self.base_path = "/mas/projects/molecularmachines/db/enzymemap/"
#         self.data = pickle.load(
#             open(os.path.join(self.base_path, "enzymemap_v2_processed.p"), "rb")
#         )
#         self.max_atoms = max_atoms
#         if max_atoms > 0:
#             self.index = [
#                 i
#                 for i in range(len(self.data))
#                 if self.data[i]["num_atoms"] <= self.max_atoms
#             ]
#         else:
#             self.index = np.arange(len(self.data))

#         self.padding = PairPad()

#         print(f"Loaded {len(self.index)} datapoints")
#         self.splits = {"train": self}

#     def __len__(self):
#         return len(self.index)

#     def get_token(self, smiles):
#         smiles = [r for r in smiles if r != "[H+]"]
#         smiles = ".".join(smiles)
#         mol = Chem.MolFromSmiles(smiles)
#         mapping_numbers = np.array(
#             [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
#         ).argsort()
#         atomic_num = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])[
#             mapping_numbers
#         ]
#         return atomic_num

#     def __getitem__(self, index):
#         entry = self.data[self.index[index]]
#         token = self.get_token(entry["mapped_reactants"])
#         reactants = entry["reacts_coords"][np.random.randint(0, 3)].squeeze()
#         products = entry["prods_coords"][np.random.randint(0, 3)].squeeze()
#         reacts_mask = entry["reacts_mol_mask"]
#         prods_mask = entry["prods_mol_mask"]
#         mask = np.ones_like(token)

#         datum = ReactDatum(
#             token=token,
#             reacts=reactants,
#             prods=products,
#             reacts_mask=reacts_mask,
#             prods_mask=prods_mask,
#             mask=mask,
#             protein_token=None,
#             protein_mask=None,
#         )

#         datum = datum.centralize()

#         if self.max_atoms > 0:
#             datum = self.padding.transform(
#                 datum,
#                 {
#                     "token": self.max_atoms,
#                     "reacts": self.max_atoms,
#                     "prods": self.max_atoms,
#                     "mask": self.max_atoms,
#                     "reacts_mask": self.max_atoms,
#                     "prods_mask": self.max_atoms,
#                 },
#             )

#         return datum


# ATOM_TYPES = {
#     "benzene": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
#     "ethanol": np.array([0, 0, 2, 1, 1, 1, 1, 1, 1]),
#     "phenol": np.array([0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1]),
#     "resorcinol": np.array([0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 1, 1, 1, 1]),
#     "ethane": np.array([0, 0, 1, 1, 1, 1, 1, 1]),
#     "malonaldehyde": np.array([2, 0, 0, 0, 2, 1, 1, 1, 1]),
# }


# class MDDensityDataset(Dataset):
#     def __init__(self, mol_name, samples=1000, _split="train", _rotated=True):
#         """
#         Density dataset for small molecules in the MD datasets.
#         Note that the validation and test splits are the same.
#         :param root: data root
#         :param mol_name: name of the molecule
#         :param split: data split, can be 'train', 'validation', 'test'
#         """
#         super(MDDensityDataset, self).__init__()
#         assert mol_name in (
#             "benzene",
#             "ethanol",
#             "phenol",
#             "resorcinol",
#             "ethane",
#             "malonaldehyde",
#         )
#         self.root = "/mas/projects/molecularmachines/db/MDDensity"
#         self.mol_name = mol_name
#         self.split = _split
#         self.n_grid = 50  # number of grid points along each dimension
#         self.grid_size = 20.0  # box size in Bohr
#         self.data_path = os.path.join(self.root, mol_name, f"{mol_name}_{self.split}")
#         self.samples = samples
#         self.atom_type = ATOM_TYPES[mol_name]
#         self.atom_coords = np.load(os.path.join(self.data_path, "structures.npy"))
#         self.densities = self._convert_fft(
#             np.load(os.path.join(self.data_path, "dft_densities.npy"))
#         )
#         self.grid_coord = self._generate_grid()
#         print("Loaded {} {} {} datapoints".format(mol_name, self.split, len(self)))
#         if _split == "train":
#             self.splits = {"train": RotatingPoolData(self, 300) if _rotated else self}
#             test = self.__class__(
#                 mol_name,
#                 samples=samples * 5,
#                 _split="test",
#             )
#             self.splits["test"] = RotatingPoolData(test, 90) if _rotated else test

#     def _convert_fft(self, fft_coeff):
#         # The raw data are stored in Fourier basis, we need to convert them back.
#         print(f"Precomputing {self.split} density from FFT coefficients ...")
#         fft_coeff = np.array(fft_coeff).astype(np.complex64)
#         d = fft_coeff.reshape(-1, self.n_grid, self.n_grid, self.n_grid)
#         hf = self.n_grid // 2
#         # first dimension
#         d[:, :hf] = (d[:, :hf] - d[:, hf:] * 1j) / 2
#         d[:, hf:] = np.flip(d[:, 1 : hf + 1], [1]).conj()
#         d = np.fft.ifft(d, axis=1)
#         # second dimension
#         d[:, :, :hf] = (d[:, :, :hf] - d[:, :, hf:] * 1j) / 2
#         d[:, :, hf:] = np.flip(d[:, :, 1 : hf + 1], [2]).conj()
#         d = np.fft.ifft(d, axis=2)
#         # third dimension
#         d[..., :hf] = (d[..., :hf] - d[..., hf:] * 1j) / 2
#         d[..., hf:] = np.flip(d[..., 1 : hf + 1], [3]).conj()
#         d = np.fft.ifft(d, axis=3)
#         return np.flip(d.real.reshape(-1, self.n_grid**3), [-1])

#     def _generate_grid(self):
#         x = np.linspace(self.grid_size / self.n_grid, self.grid_size, self.n_grid)
#         return np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)

#     def __getitem__(self, item):
#         grid = self.grid_coord
#         density = self.densities[item]

#         if self.samples:
#             probes = np.random.randint(0, len(grid), self.samples)
#             grid = grid[probes]
#             density = density[probes]

#         datum = DensityDatum(
#             density=density.astype(np.float32),
#             grid=grid.astype(np.float32),
#             atom_coord=self.atom_coords[item].astype(np.float32),
#             atom_token=self.atom_type.astype(np.int32),
#             atom_mask=np.ones_like(self.atom_type),
#             bonds=None,
#         )
#         return datum

#     def __len__(self):
#         return self.atom_coords.shape[0]


# import lzma


# class CubeDataset(Dataset):
#     def __init__(self, max_atoms=10, samples=1000, _split="train", _rotated=True):
#         """
#         The density dataset contains volumetric data of molecules.
#         :param root: data root
#         :param split: data split, can be 'train', 'validation', 'test'
#         :param split_file: the data split file containing file names of the split
#         """
#         super(CubeDataset).__init__()
#         self.root = "/mas/projects/molecularmachines/db/CubeDenisty"
#         self.split = _split
#         self.extension = "json"
#         self.compression = "xz"

#         self.file_pattern = f".{self.extension}"
#         self.file_pattern += f".{self.compression}"

#         # with open(os.path.join(self.root, split_file)) as f:
#         # reverse the order so that larger molecules are tested first
#         # self.file_list = list(reversed(json.load(f)[split]))

#         self.file_list = list(reversed(os.listdir(self.root)))
#         try:
#             self.file_list.remove("crystal.json")
#             self.file_list.remove("num_atoms.npy")
#         except ValueError:
#             pass

#         self.num_atoms = np.load(os.path.join(self.root, "num_atoms.npy"))
#         self.max_atoms = max_atoms
#         if self.max_atoms:
#             self.file_list = np.array(self.file_list)[
#                 np.where(self.num_atoms <= max_atoms)
#             ]

#         np.random.seed(0)
#         if _split == "train":
#             self.file_list = np.random.permutation(self.file_list)[:-2000]
#         elif _split == "test":
#             self.file_list = np.random.permutation(self.file_list)[-2000:-1000]
#         elif _split == "valid":
#             self.file_list = np.random.permutation(self.file_list)[-1000:]

#         with open(os.path.join(self.root, "crystal.json")) as f:
#             atom_info = json.load(f)
#         atom_list = [info["name"] for info in atom_info]
#         self.atom_name2idx = {name: idx for idx, name in enumerate(atom_list)}
#         self.atom_name2idx.update(
#             {name.encode(): idx for idx, name in enumerate(atom_list)}
#         )
#         self.atom_num2idx = {
#             info["atom_num"]: idx for idx, info in enumerate(atom_info)
#         }
#         self.idx2atom_num = {
#             idx: info["atom_num"] for idx, info in enumerate(atom_info)
#         }
#         self.padding = PairPad()

#         self.open = lzma.open
#         self.samples = samples
#         self.elements = elements.assign(
#             symbol=lambda df: df.symbol.str.lower()
#         ).symbol.tolist()

#         print("Loaded {} {} datapoints".format(self.split, len(self)))
#         if _split == "train":
#             self.splits = {"train": RotatingPoolData(self, 300) if _rotated else self}
#             test = self.__class__(
#                 samples=samples,
#                 max_atoms=max_atoms,
#                 _split="test",
#             )
#             self.splits["test"] = RotatingPoolData(test, 90) if _rotated else test
#             valid = self.__class__(
#                 samples=samples,
#                 max_atoms=max_atoms,
#                 _split="valid",
#             )
#             self.splits["valid"] = RotatingPoolData(test, 30) if _rotated else valid

#     def __getitem__(self, item):
#         atom_type, atom_coord, density, grid_coord, info = self.read_json(item)

#         if self.samples:
#             probes = np.random.randint(0, len(grid_coord), self.samples)
#             grid_coord = grid_coord[probes]
#             density = density[probes]

#         datum = DensityDatum(
#             density=density.astype(np.float32),
#             grid=grid_coord.astype(np.float32),
#             atom_coord=atom_coord.astype(np.float32),
#             atom_token=atom_type.astype(np.int32),
#             atom_mask=np.ones_like(atom_type),
#             bonds=None,
#         )

#         if self.max_atoms:
#             datum = self.padding.transform(
#                 datum,
#                 {
#                     "atom_token": self.max_atoms,
#                     "atom_coord": self.max_atoms,
#                     "atom_mask": self.max_atoms,
#                 },
#             )

#         return datum

#     def __len__(self):
#         return len(self.file_list)

#     def read_json(self, item):
#         """Read atoms and data from JSON file."""

#         def read_2d_tensor(s):
#             return np.array([[float(x) for x in line] for line in s])

#         with self.open(os.path.join(self.root, self.file_list[item])) as fileobj:
#             data = json.load(fileobj)

#         scale = float(data["vector"][0][0])
#         cell = read_2d_tensor(data["lattice"][0]) * scale
#         elements = data["elements"][0]
#         n_atoms = [int(s) for s in data["elements_number"][0]]

#         tot_atoms = sum(n_atoms)
#         atom_coord = read_2d_tensor(data["coordinates"][0])
#         atom_type = np.zeros(tot_atoms, dtype=np.int32)
#         idx = 0
#         for elem, n in zip(elements, n_atoms):
#             atom_type[idx : idx + n] = self.elements.index(elem.lower()) + 1
#             # atom_type[idx : idx + n] = self.idx2atom_num[self.atom_name2idx[elem]]
#             idx += n

#         atom_coord = atom_coord @ cell

#         shape = [int(s) for s in data["FFTgrid"][0]]
#         x_coord = np.linspace(0, shape[0] - 1, shape[0])[..., None] / shape[0] * cell[0]
#         y_coord = np.linspace(0, shape[1] - 1, shape[1])[..., None] / shape[1] * cell[1]
#         z_coord = np.linspace(0, shape[2] - 1, shape[2])[..., None] / shape[2] * cell[2]
#         grid_coord = (
#             x_coord.reshape(-1, 1, 1, 3)
#             + y_coord.reshape(1, -1, 1, 3)
#             + z_coord.reshape(1, 1, -1, 3)
#         )
#         grid_coord = grid_coord.reshape(-1, 3)

#         n_grid = shape[0] * shape[1] * shape[2]
#         n_line = (n_grid + 9) // 10
#         density = np.array(
#             [
#                 float(s) if not s[0] == "*" else 0.0
#                 for line in data["chargedensity"][0][:n_line]
#                 for s in line
#             ]
#         ).reshape(-1)[:n_grid]
#         volume = np.abs(np.linalg.det(cell))
#         density = density / volume
#         density = (
#             density.reshape(shape[2], shape[1], shape[0]).transpose(2, 1, 0).reshape(-1)
#         )

#         return (
#             atom_type,
#             atom_coord,
#             density,
#             grid_coord,
#             {"shape": shape, "cell": cell},
#         )
