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
    StandardizeProperties,
    SortAtoms,
    PairPad,
)
from .utils import pids_file_to_list, extract_rdkit_mol_properties
from .alphabet import PERIODIC_TABLE, elements
from rdkit import Chem


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
        self.padding = MoleculePad(_max_atoms) if _padding else False
        self.permute = Permuter() if permute else None
        self.centralize = Centralize() if centralize else None
        self.sorter = SortAtoms()
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

        datum = self.sorter.transform(datum)

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


import lmdb
from moleculib.molecule.datum import CrossdockDatum


class CrossdockDataset(Dataset):

    def __init__(self, max_ligand_atoms=29, max_protein_atoms=400, _split="train"):
        super().__init__()

        base_path = "/mas/projects/molecularmachines/db/crossdocked_targetdiff/"
        self.processed_path = os.path.join(
            base_path, "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
        )
        self.split_file = os.path.join(base_path, "crossdocked_pocket10_pose_split.pkl")
        self.index_file = os.path.join(base_path, "crossdocked_pocket10_pose_index.pkl")
        self.db = None

        self.padding = PairPad()
        self.max_ligand_atoms = max_ligand_atoms
        self.max_protein_atoms = max_protein_atoms

        with open(self.index_file, "rb") as f:
            data = pickle.load(f)
        ligand_atoms = data["ligand_atoms"]
        ligand_keys = set()
        protein_atoms = data["protein_atoms"]
        protein_keys = set()
        for key, value in ligand_atoms:
            if value <= self.max_ligand_atoms or self.max_ligand_atoms == -1:
                ligand_keys.add(key)
            else:
                break
        for key, value in protein_atoms:
            if value <= self.max_protein_atoms or self.max_protein_atoms == -1:
                protein_keys.add(key)
            else:
                break
        with open(self.split_file, "rb") as f:
            split_keys = set(pickle.load(f)[_split])
        self.keys = ligand_keys & protein_keys & split_keys
        self.keys = list(self.keys)
        print(f"For {_split} loaded {len(self.keys)} pairs")
        if _split == "train":
            self.splits = {
                "train": self,
                "val": self.__class__(
                    self.max_ligand_atoms, self.max_protein_atoms, _split="val"
                ),
                "test": self.__class__(
                    self.max_ligand_atoms, self.max_protein_atoms, _split="test"
                ),
            }
            self.splits = {k: v for k, v in self.splits.items() if len(v) > 0}

    def _connect_db(self):
        """
        Establish read-only database connection
        """
        assert self.db is None, "A connection has already been opened."
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        data, key = self.get_ori_data(idx)
        filename = data["ligand_filename"]
        atom_token = np.array(data["ligand_element"])
        atom_coord = np.array(data["ligand_pos"])
        bonds = data["ligand_bond_index"]
        bonds = np.row_stack((bonds, data["ligand_bond_type"])).T
        atom_features = np.array(data["ligand_atom_feature"])

        protein_token = np.array(data["protein_element"])
        protein_coord = np.array(data["protein_pos"])

        datum = CrossdockDatum(
            key=key,
            filename=filename,
            atom_token=atom_token,
            atom_coord=atom_coord,
            atom_mask=np.ones_like(atom_token),
            bonds=bonds,
            atom_features=atom_features,
            protein_token=protein_token,
            protein_coord=protein_coord,
            protein_mask=np.ones_like(protein_token),
        )

        datum = self.padding.transform(
            datum,
            {
                "atom_token": self.max_ligand_atoms,
                "atom_coord": self.max_ligand_atoms,
                "atom_mask": self.max_ligand_atoms,
                "bonds": self.max_ligand_atoms,
                "atom_features": self.max_ligand_atoms,
                "protein_token": self.max_protein_atoms,
                "protein_coord": self.max_protein_atoms,
                "protein_mask": self.max_protein_atoms,
            },
        )
        return datum

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(eval(f"b'{key}'")))
        return data, key


import biotite.structure.io.pdb as pdb
from moleculib.molecule.datum import PDBBindDatum
import biotite.structure.io as strucio
import pickle


class PDBBindDataset(Dataset):
    def __init__(self, max_ligand_atoms=29, max_protein_atoms=400, _split="train"):
        super().__init__()
        self.max_ligand_atoms = max_ligand_atoms
        self.max_protein_atoms = max_protein_atoms
        self.base_path = "/mas/projects/molecularmachines/db/PDBBind/refined-set"
        self.index_path = os.path.join(self.base_path, "index/INDEX_refined_data.2020")
        self.index = self._load_index(_split)
        print(f"Loaded {self.index.shape[0]} {_split} datapoints")

        self.padding = PairPad()
        self.elements = elements.assign(
            symbol=lambda df: df.symbol.str.lower()
        ).symbol.tolist()  # TODO:

        if _split == "train":
            self.splits = {
                "train": self,
                # "val": self.__class__(
                #     self.max_ligand_atoms, self.max_protein_atoms, _split="val"
                # ),
                "test": self.__class__(
                    self.max_ligand_atoms, self.max_protein_atoms, _split="test"
                ),
            }
            self.splits = {k: v for k, v in self.splits.items() if len(v) > 0}

    def _load_index(self, split):
        KMAP = {"Ki": 1, "Kd": 2, "IC50": 3}

        all_files = os.listdir(self.base_path)
        all_index = []
        with open(self.index_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            index, res, year, pka, kv = line.split("//")[0].strip().split()

            kind = [v for k, v in KMAP.items() if k in kv]
            assert len(kind) == 1
            if index in all_files:
                all_index.append([index, res, year, pka, kind[0]])

        all_index = np.array(all_index)
        with open(os.path.join(self.base_path, "index/lengths.pkl"), "rb") as f:
            atoms_count = pickle.load(f)
            prot_len = np.array(atoms_count["prot_len"]).squeeze()
            lig_len = np.array(atoms_count["lig_len"]).squeeze()
        protm = prot_len <= self.max_protein_atoms
        ligm = lig_len <= self.max_ligand_atoms
        sub_index = all_index[protm & ligm]

        with open(f"{self.base_path}/timesplit_test", "r") as f:
            test_split = [l.strip("\n") for l in f.readlines()]
        split_index = []
        for d in sub_index:
            if split == "train" and d[0] in test_split:
                continue
            if split == "test" and d[0] not in test_split:
                continue
            split_index.append(d)

        return np.array(split_index)

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, idx):
        pdb_id, _, _, pka, _ = self.index[idx]
        pka = float(pka)
        protein_coord, protein_token = self._get_protein_pocket(pdb_id)
        atom_coord, atom_token, bonds, charge = self._get_ligand(pdb_id)

        datum = PDBBindDatum(
            pdb_id=pdb_id,
            pka=np.array(pka),
            atom_token=atom_token,
            atom_coord=atom_coord,
            atom_mask=np.ones_like(atom_token),
            charge=charge,
            bonds=bonds,
            protein_coord=protein_coord,
            protein_token=protein_token,
            protein_mask=np.ones_like(protein_token),
        )

        datum = self.padding.transform(
            datum,
            {
                "atom_token": self.max_ligand_atoms,
                "atom_coord": self.max_ligand_atoms,
                "atom_mask": self.max_ligand_atoms,
                "charge": self.max_ligand_atoms,
                "bonds": self.max_ligand_atoms,
                "protein_token": self.max_protein_atoms,
                "protein_coord": self.max_protein_atoms,
                "protein_mask": self.max_protein_atoms,
            },
        )
        return datum

    def _get_protein_pocket(self, pdb_id):
        filepath = os.path.join(self.base_path, pdb_id, f"{pdb_id}_pocket.pdb")
        pdb_file = pdb.PDBFile.read(filepath)
        atom_array = pdb.get_structure(pdb_file, model=1)
        coord = atom_array.coord
        token = [self.elements.index(e.lower()) + 1 for e in atom_array.element]

        return np.array(coord), np.array(token)

    def _get_ligand(self, pdb_id):
        filepath = os.path.join(self.base_path, pdb_id, f"{pdb_id}_ligand.sdf")
        mol = strucio.load_structure(filepath)
        token = [self.elements.index(e.lower()) + 1 for e in mol.element]
        return (
            mol.coord,
            np.array(token),
            mol.bonds._bonds.astype(np.int32),
            mol.charge.astype(np.int32),
        )


from ase.calculators.vasp import VaspChargeDensity
import lz4.frame
import tempfile


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


from moleculib.molecule.datum import DensityDatum
import random


class DensityDataDir(Dataset):
    def __init__(
        self,
        directory,
        max_atoms=29,
        grid_size=36,
        to_split=True,
        _split="train",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.directory = directory
        self.padding = PairPad()
        self.max_atoms = max_atoms
        self.grid_size = grid_size
        self.member_list = []
        for s in [
            # "0",
            # "2",
            # "3",
            # "4",
            # "5",
            "6",
            "7",
            "8",
            "9",
        ]:
            self.member_list.extend(sorted(os.listdir(os.path.join(self.directory, s))))
            # self.member_list.extend(
            #     sorted(
            #         [
            #             f
            #             for f in os.listdir(os.path.join(self.directory, s))
            #             if f.endswith(".npz")
            #         ]
            #     )
            # )

        # random.shuffle(self.member_list)

        dp = len(self.member_list)
        if to_split:
            if _split == "train":
                self.member_list = self.member_list[: round(dp * 0.95)]
            elif _split == "test":
                self.member_list = self.member_list[round(dp * 0.95) :]

        print(f"Loaded {_split} {len(self)} datapoints")

        if _split == "train":
            self.splits = {"train": self}
            if to_split:
                self.splits["test"] = self.__class__(
                    directory=directory,
                    max_atoms=max_atoms,
                    grid_size=grid_size,
                    _split="test",
                )

    def __len__(self):
        return len(self.member_list)

    def extractfile(self, index):
        filename = self.member_list[index]
        path = os.path.join(
            self.directory, f"{int(filename.split('.')[0]) // 1000}", filename
        )
        if filename.endswith(".npz"):
            with np.load(path) as f:
                return {
                    "density": f["density"],
                    "coord": f["coord"],
                    "token": f["token"],
                    "grid": f["grid"],
                    "filename": filename,
                }

        filecontent = _decompress_file(path)
        density, atoms, origin = _read_vasp(filecontent)

        grid_pos = _calculate_grid_pos(density, origin, atoms.get_cell())

        return {
            "density": density,
            "coord": atoms.positions,
            "token": atoms.numbers,
            "grid": grid_pos,
            "filename": filename,
        }

    def _process(self, index):
        dp = self.extractfile(index)
        file_num = dp["filename"].split(".")[0]
        path = os.path.join(
            self.directory, f"{int(file_num) // 1000}", file_num + ".npz"
        )
        np.savez(
            path,
            density=dp["density"],
            grid=dp["grid"],
            coord=dp["coord"],
            token=dp["token"],
        )

    def __getitem__(self, index):
        dp = self.extractfile(index)

        density = dp["density"]
        grid = dp["grid"]

        assert (
            density.shape[0] >= self.grid_size
        ), f"{density.shape[0]} < {self.grid_size}, idx {index}"
        assert (
            density.shape[1] >= self.grid_size
        ), f"{density.shape[1]} < {self.grid_size}, idx {index}"
        assert (
            density.shape[2] >= self.grid_size
        ), f"{density.shape[2]} < {self.grid_size}, idx {index}"

        if self.grid_size:
            center = self.com(density)
            center += np.random.randint(-self.grid_size // 2, self.grid_size // 2, 3)
            x_min, x_max, y_min, y_max, z_min, z_max = self.neighborhood(
                center, density, self.grid_size
            )
            density = density[
                x_min:x_max,
                y_min:y_max,
                z_min:z_max,
            ].reshape(-1)
            grid = grid[
                x_min:x_max,
                y_min:y_max,
                z_min:z_max,
            ].reshape((-1, 3))

        datum = DensityDatum(
            density=density,
            grid=grid,
            atom_coord=dp["coord"],
            atom_token=dp["token"],
            atom_mask=np.ones_like(dp["token"]),
            bonds=None,
        )

        datum = self.padding.transform(
            datum,
            {
                "atom_token": self.max_atoms,
                "atom_coord": self.max_atoms,
                "atom_mask": self.max_atoms,
            },
        )

        return datum

    def com(self, density_array):
        # Create arrays of indices along each axis
        x_indices, y_indices, z_indices = np.indices(density_array.shape)

        # Calculate the total mass
        total_mass = np.sum(density_array)

        # Calculate the center of mass along each axis
        center_x = np.sum(x_indices * density_array) / total_mass
        center_y = np.sum(y_indices * density_array) / total_mass
        center_z = np.sum(z_indices * density_array) / total_mass

        return np.round([center_x, center_y, center_z]).astype(np.int32)

    def neighborhood(self, center, density_array, n):
        # Calculate the center of mass indices
        center_x, center_y, center_z = center

        # Calculate the boundaries for indexing
        x_min = max(0, center_x - n // 2)
        x_max = min(density_array.shape[0], center_x + n // 2)
        y_min = max(0, center_y - n // 2)
        y_max = min(density_array.shape[1], center_y + n // 2)
        z_min = max(0, center_z - n // 2)
        z_max = min(density_array.shape[2], center_z + n // 2)

        if (x_max - x_min) < n:
            if x_max == density_array.shape[0]:
                x_min -= n - (x_max - x_min)
            else:
                x_max += n - (x_max - x_min)
        if (y_max - y_min) < n:
            if y_max == density_array.shape[1]:
                y_min -= n - (y_max - y_min)
            else:
                y_max += n - (y_max - y_min)
        if (z_max - z_min) < n:
            if z_max == density_array.shape[2]:
                z_min -= n - (z_max - z_min)
            else:
                z_max += n - (z_max - z_min)

        return x_min, x_max, y_min, y_max, z_min, z_max


import h5py
from moleculib.molecule.datum import MISATODatum

class MISATO(Dataset):
    def __init__(self, _split="train") -> None:
        super().__init__()
        self.base_path = "/mas/projects/molecularmachines/db/MISATO"
        self.data = h5py.File("/mas/projects/molecularmachines/db/MISATO/MD.hdf5")
        self.h5_properties = [
            "trajectory_coordinates",
            # "atoms_type",
            "atoms_number",
            # "atoms_residue",
            # "atoms_element",
            "molecules_begin_atom_index",
            # "frames_rmsd_ligand",
            # "frames_distance",
            # "frames_interaction_energy",
            # "frames_bSASA",
        ]
        if _split == "train":
            self.index = open(os.path.join(self.base_path,"train_MD.txt"), 'r').read().split('\n')
        elif _split == "val":
            self.index = open(os.path.join(self.base_path,"val_MD.txt"), 'r').read().split('\n')
        elif _split == "test":
            self.index = open(os.path.join(self.base_path,"test_MD.txt"), 'r').read().split('\n')
        
        print(f"Loaded {_split} {len(self)} datapoints")

        if _split == "train":
            self.splits = {"train": self}
            self.splits["val"] = self.__class__(
                _split="val",
            )
            self.splits["test"] = self.__class__(
                _split="test",
            )

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        pdb_id = self.index[index]
        dp = self.get_entries(pdb_id)

        traj_coord = dp["trajectory_coordinates"]
        token = dp["atoms_number"]
        mol_idx = dp["molecules_begin_atom_index"][-1]

        atom_token = token[mol_idx:]
        atom_coord = traj_coord[:, mol_idx:]
        atom_mask = np.ones_like(atom_token)
        
        protein_token = token[:mol_idx]
        protein_coord = traj_coord[:, :mol_idx]
        protein_mask = np.ones_like(protein_token)
        
        datum = MISATODatum(
            pdb_id=pdb_id,
            atom_token=atom_token,
            atom_coord=atom_coord,
            atom_mask=atom_mask,
            bonds=None,
            protein_token=protein_token,
            protein_coord=protein_coord,
            protein_mask=protein_mask,
        )
        return datum
    
    def get_entries(self, pdbid):
        h5_entries = {}
        for h5_property in self.h5_properties:
            h5_entries[h5_property] = self.data.get(pdbid+'/'+h5_property)
        return h5_entries