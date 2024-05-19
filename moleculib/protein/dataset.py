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
from tqdm.contrib.concurrent import process_map

# For Chignolin
import mdtraj
import torch.utils.data
import biotite.structure.io.pdb as pdb

from .alphabet import UNK_TOKEN
from .datum import ProteinDatum
from .transform import ProteinTransform
from .utils import pids_file_to_list

from ..abstract.dataset import PreProcessedDataset

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


class PDBDataset:
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
        keep_ids: List[str] = None,
        filter_ids=None,
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

        if keep_ids is not None:
            self.metadata = self.metadata[self.metadata["idcode"].isin(keep_ids)]

        # shuffle and sample
        self.metadata = self.metadata.sample(frac=frac).reset_index(drop=True)
        print(f"Loaded metadata with {len(self.metadata)} samples")

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

        self.filter_ids = (
            [i.lower() for i in filter_ids] if filter_ids is not None else None
        )

    def _is_in_filter(self, sample):
        return int(sample["id"]) in self.shard_indices

    def __len__(self):
        return len(self.metadata)

    def load_index(self, idx):
        while True:
            header = self.metadata.iloc[idx]
            pdb_id = header["idcode"]
            if self.filter_ids is not None:
                if pdb_id.lower() in self.filter_ids:
                    idx = np.random.randint(0, len(self.metadata))
                    continue
            break

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
    def _maybe_fetch_and_extract(pdb_id, format, save_path):
        try:
            if os.path.exists(os.path.join(save_path, f"{pdb_id}.{format}")):
                return None
            datum = ProteinDatum.fetch_pdb_id(
                pdb_id, save_path=save_path, format=format
            )
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
        format: str = "mmtf",
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
            save_path = mkdtemp()
        print(f"Fetching {len(pdb_ids)} PDB IDs with {max_workers} workers...")

        series = {c: Series(dtype=t) for (c, t) in PDB_METADATA_FIELDS}
        metadata = DataFrame(series)

        extractor = partial(
            cls._maybe_fetch_and_extract, save_path=save_path, format=format
        )
        if max_workers > 1:
            extraction = process_map(extractor, pdb_ids, max_workers=max_workers)
        else:
            extraction = list(map(extractor, pdb_ids))
        extraction = filter(lambda x: x, extraction)
        _, metadata_ = list(map(list, zip(*extraction)))
        metadata = pd.concat((metadata, *metadata_), axis=0)

        if save:
            with open(str(Path(save_path) / "metadata.pyd"), "wb") as file:
                pickle.dump(metadata, file)

        return cls(base_path=save_path, metadata=metadata, **kwargs)


class MonomerDataset(PDBDataset):
    def __init__(
        self,
        base_path: str,
        metadata: pd.DataFrame = None,
        single_chain: bool = True,
        **kwargs,
    ):
        # read from base path if metadata is not built
        if metadata is None:
            with open(str(Path(base_path) / "metadata.pyd"), "rb") as file:
                metadata = pickle.load(file)
        metadata = metadata.reset_index()

        # NOTE(Allan): small hack to make sure
        # we follow trainer.py convention
        self.splits = {"train": self}

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
        # if single_chain:
        # breakpoint()
        # metadata = metadata[::MAX_COMPLEX_SIZE].reset_index()

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


from typing import Callable


class TinyPDBDataset(PreProcessedDataset):

    def __init__(self, base_path, transform: List[Callable] = None, shuffle=True):
        base_path = os.path.join(base_path, "tinypdb.pyd")
        with open(base_path, "rb") as fin:
            print("Loading data...")
            splits = pickle.load(fin)
        super().__init__(splits, transform, shuffle, pre_transform=False)


class FrameDiffDataset(PreProcessedDataset):

    def __init__(self, base_path, transform: List[Callable] = None, shuffle=True):
        base_path = os.path.join(base_path, "framediff_train_data.pyd")
        with open(base_path, "rb") as fin:
            print("Loading data...")
            splits = pickle.load(fin)
        super().__init__(splits, transform, shuffle, pre_transform=False)


class TinyPDBDataset(PreProcessedDataset):

    def __init__(self, base_path, transform: List[Callable] = None, shuffle=True):
        base_path = os.path.join(base_path, "tinypdb.pyd")
        with open(base_path, "rb") as fin:
            print("Loading data...")
            splits = pickle.load(fin)
        super().__init__(splits, transform, shuffle, pre_transform=False)


class FoldingDiffDataset(PreProcessedDataset):

    def __init__(self, base_path, transform: List[Callable] = None, shuffle=True):
        base_path = os.path.join(base_path, "folddiff_train_data.pyd")
        with open(base_path, "rb") as fin:
            print("Loading data...")
            splits = pickle.load(fin)
        super().__init__(splits, transform, shuffle, pre_transform=False)


class FoldDataset(PreProcessedDataset):

    def __init__(self, base_path, transform: List[Callable] = None, shuffle=True):
        base_path = os.path.join(base_path, "fold.pyd")
        with open(base_path, "rb") as fin:
            print("Loading data...")
            splits = pickle.load(fin)
        super().__init__(splits, transform, shuffle)


class EnzymeCommissionDataset(PreProcessedDataset):

    def __init__(self, base_path, transform: List[Callable] = None, shuffle=True):
        path = os.path.join(base_path, "ec.pyd")
        with open(path, "rb") as fin:
            print(f"Loading data from {path}")
            splits = pickle.load(fin)
        super().__init__(splits, transform, shuffle)


class GeneOntologyDataset(PreProcessedDataset):

    def __init__(
        self, base_path, transform: List[Callable] = None, level="mf", shuffle=True
    ):
        path = os.path.join(base_path, f"go_{level}.pyd")
        with open(path, "rb") as fin:
            print(f"Loading data from {path}")
            splits = pickle.load(fin)
        super().__init__(splits, transform, shuffle)


class FuncDataset(PreProcessedDataset):

    def __init__(self, base_path, transform: List[Callable] = None, shuffle=True):
        path = os.path.join(base_path, "func.pyd")
        with open(path, "rb") as fin:
            print(f"Loading data from {path}")
            splits = pickle.load(fin)
        super().__init__(splits, transform, shuffle)


class ScaffoldsDataset(PreProcessedDataset):

    def __init__(
        self, base_path, transform: List[Callable] = None, shuffle=True, val_split=0.0
    ):
        with open(os.path.join(base_path, "scaffolds.pyd"), "rb") as fin:
            print("Loading data...")
            dataset = pickle.load(fin)
        if val_split > 0.0:
            print(f"Splitting data into train/val with val_split={val_split}")
            dataset = np.random.permutation(dataset)
            num_val = int(len(dataset) * val_split)
            splits = dict(train=dataset[:-num_val], val=dataset[-num_val:])
        else:
            splits = dict(train=dataset)
        super().__init__(splits, transform, shuffle)




class FastFoldingDataset(torch.utils.data.Dataset):
    """The FastFoldingProteins dataset from the Two-For-One paper."""

    def __init__(
        self, protein="chignolin", num_files=-1, tau=1, stride=1, time_sort=False
    ):
        base = "/mas/projects/molecularmachines/db/FastFoldingProteins/"
        if protein == "chignolin":
            self.base_path = base + "chignolin_trajectories/filtered/"
        elif "trpcage" in protein: # trpcage0, trpcage1, trpcage2
            self.base_path = base + f"rpcage_trajectories/batches/{protein[-1]}/filtered"
        elif protein == "villin":
            self.base_path = base + "villin_trajectories/filtered/"
        elif "bba" in protein: # bba0, bba1, bba2
            self.base_path = base + f"bba_trajectories/batches/{protein[-1]}/filtered"

        self.num_files = num_files
        self.tau = tau
        self.stride = stride
        self.time_sort = time_sort
        self.files = self._list_files()[: self.num_files]
        self.atom_array = pdb.PDBFile.read(
            self.base_path + "filtered.pdb"
        ).get_structure()[0]
        self.aa_filter = biotite.structure.filter_amino_acids(self.atom_array)
        self.atom_array = self.atom_array[self.aa_filter]
        self.counter = 0
        self._load_coords(self.files[0])
        print(f"{len(self)} total samples")

    def _list_files(self):
        def extract_x_y(filename):
            part = os.path.basename(filename).split("_")[0]
            x, y = part.strip("e").split("s")
            return int(x), int(y)

        files_with_extension = set()
        for filename in os.listdir(self.base_path):
            if filename.endswith(".xtc") and not filename.startswith("."):
                files_with_extension.add(self.base_path + filename)

        files = list(files_with_extension)
        if self.time_sort:
            return sorted(files, key=lambda x: extract_x_y(x))
        return files

    def _load_coords(self, files):
        data = mdtraj.load(
            files,
            top=self.base_path + "filtered.pdb",
            stride=self.stride,
        )
        self.coords = data.xyz[:, self.aa_filter, :] * 10  # Convert to angstroms

    def _num_timesteps(self):
        return self.coords.shape[0]

    def __len__(self):
        return len(self.files) * self._num_timesteps()

    def __getitem__(self, idx):
        if self.counter > 100:
            self._load_coords(self.files[idx // self._num_timesteps()])
            self.counter = 0
        self.counter += 1
        idxx = idx % (self._num_timesteps() - self.tau)

        self.atom_array._coord = self.coords[idxx]
        p1 = ProteinDatum.from_atom_array(
            self.atom_array,
            header=dict(
                idcode=None,
                resolution=None,
            ),
        )
        self.atom_array._coord = self.coords[idxx + self.tau]
        p2 = ProteinDatum.from_atom_array(
            self.atom_array,
            header=dict(
                idcode=None,
                resolution=None,
            ),
        )
        return [p2, p1]



class TimewarpDataset(torch.utils.data.Dataset):
    """Exposes datasets from the Timewarp paper."""

    def __init__(
        self,
        dataset: str = "2AA-1-big",
        split: str = "train",
        tau: int = 1,
    ):
        base = "/mas/projects/molecularmachines/db/timewarp/"
        self.base_path = os.path.join(base, dataset, split)
        self.counter = 0
        self.tau = tau

        self.files = self._list_files()
        print(f"Found {len(self.files)} files in {self.base_path}")

        self._load_coords(self.files[0])

    def _list_files(self):
        files_with_extension = set()
        for filename in os.listdir(self.base_path):
            if filename.endswith(".npz") and not filename.startswith("."):
                files_with_extension.add(os.path.join(self.base_path, filename))
        return list(files_with_extension)

    def _load_coords(self, file):
        data = np.load(file)
        pdb_file = file.replace("-arrays.npz", "-state0.pdb")
        self.atom_array = pdb.PDBFile.read(pdb_file).get_structure()[0]
        self.coords = data['positions'] * 10  # Convert to angstroms

    def __len__(self):
        return len(self.files) * self.coords.shape[0]

    def _num_timesteps(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        if self.counter > 1000:
            self._load_coords(self.files[idx // self._num_timesteps()])
            self.counter = 0
        self.counter += 1
        idxx = idx % (self._num_timesteps() - self.tau)

        self.atom_array._coord = self.coords[idxx]
        p1 = ProteinDatum.from_atom_array(
            self.atom_array,
            header=dict(
                idcode=None,
                resolution=None,
            ),
        )
        self.atom_array._coord = self.coords[idxx + self.tau]
        p2 = ProteinDatum.from_atom_array(
            self.atom_array,
            header=dict(
                idcode=None,
                resolution=None,
            ),
        )
        return [p2, p1]