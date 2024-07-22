from collections import defaultdict
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

from tqdm import tqdm

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



FAST_FOLDING_PROTEINS = {
    'chignolin': 10,
    'trpcage': 20,
    'bba': 28,
    'wwdomain': 34,
    'villin': 35,
    'ntl9': 39,
    'bbl': 47,
    'proteinb': 47,
    'homeodomain': 54,
    'proteing': 56,
    'a3D': 73,
    'lambda': 80
}


from moleculib.protein.datum import ProteinDatum

from moleculib.protein.transform import ProteinPad
from biotite.structure import filter_amino_acids
import biotite.structure.io.pdb as pdb
import numpy as np
from tqdm import tqdm

import mdtraj
import os 
from collections import defaultdict
from copy import deepcopy
from biotite.structure.io.xtc import XTCFile
from numpy.lib.format import open_memmap


class FastFoldingDataset:
    
    def __init__(
        self, 
        base = "/mas/projects/molecularmachines/db/FastFoldingProteins/memmap/",
        proteins=None, 
        tau=0,
        shuffle=True,
        stride=1,
        preload=False,
        epoch_size=10000,
        padded=True,
        num_folders=1,
    ):
        if proteins == None:
            proteins = list(FAST_FOLDING_PROTEINS.keys())

        self.base_path = base 
        self.proteins = proteins 
        self.tau = tau
        self.time_sort = True
        self.stride = stride
        self.epoch_size = epoch_size * len(proteins)
        self.num_folders = num_folders
        
        self.describe()

        self.atom_arrays = {
            protein: pdb.PDBFile.read(
                self.base_path + protein + "/0/filtered.pdb"
            ).get_structure()[0] for protein in proteins
        }
        self.atom_arrays = {
            protein: atom_array[filter_amino_acids(atom_array)] for protein, atom_array in self.atom_arrays.items()
        }

        if padded: self.pad = ProteinPad(pad_size=max([FAST_FOLDING_PROTEINS[protein] for protein in proteins]))
        else: self.pad = lambda x: x

        self.splits = { 'train': self }

    def describe(self):
        # file is indexed by protein, trajectory, and frame
        self.files = defaultdict(lambda: defaultdict(list))
        self.num_trajectories = defaultdict(int)
        self.num_frames_per_traj = defaultdict(lambda: defaultdict(int))
        self.num_frames = defaultdict(int)

        for protein in self.proteins:
            protein_path = self.base_path + protein + "/"
            for idx, trajectory in enumerate(os.listdir(protein_path)):
                if trajectory.startswith("."): continue
                self.num_trajectories[protein] += 1
                trajectory_path = protein_path + trajectory + "/"
                for supframe in os.listdir(trajectory_path):
                    if supframe.startswith("."): continue
                    if not supframe.endswith('.mmap'): continue
                    self.num_frames[protein] += 1
                    self.num_frames_per_traj[protein][int(trajectory)] += 1
                    if self.files[protein].get(int(trajectory)) is None:
                        self.files[protein][int(trajectory)] = []
                    self.files[protein][int(trajectory)].append(trajectory_path + supframe)
                if idx == self.num_folders - 1:
                    break
                    
        
        for protein in self.proteins:
            print(f"{protein}: {self.num_trajectories[protein]} trajectories, {self.num_frames[protein]} total frames")
            # print(f"Trajectory lengths: {self.num_frames_per_traj[protein]}")

    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, idx):
        protein = self.proteins[idx % len(self.proteins)]
        
        # need to check if len(coord) > self.tau + 1
        while True:
            traj_idx = np.random.randint(0, self.num_trajectories[protein])
            subtraj_idx = np.random.randint(0, self.num_frames_per_traj[protein][traj_idx])
            
            mmap_path = self.files[protein][traj_idx][subtraj_idx]
            coord = open_memmap(mmap_path, mode='r', dtype=np.float32)       
            template = self.atom_arrays[protein]
            
            if len(coord) > self.tau + 1: break
    
        idx1 = np.random.randint(0, len(coord) - self.tau - 1)

        aa1 = deepcopy(template)    
        aa1._coord = coord[idx1]

        p1 = ProteinDatum.from_atom_array(
            aa1,
            header=dict(
                idcode=protein,
                resolution=None,
            ),
        )
        
        p1 = self.pad.transform(p1)
        if self.tau == 0:
            return p1

        idx2 = idx1 + self.tau
        aa2 = deepcopy(template)
        aa2._coord = coord[idx2]
        p2 = ProteinDatum.from_atom_array(
            aa2,
            header=dict(
                idcode=protein,
                resolution=None,
            ),
        )

        p2 = self.pad.transform(p2)
        return [p1, p2]
    





from biotite.structure.io import pdb
from moleculib.protein.transform import ProteinPad


TAUS = [0, 1, 2, 4, 8, 16]
import webdataset as wds 

from copy import deepcopy
from moleculib.protein.datum import ProteinDatum


class ShardedFastFoldingDataset:
    
    def __init__(
        self, 
        base = "/mas/projects/molecularmachines/db/FastFoldingProteins/web/",
        proteins=None, 
        tau=0,
        padded=True,
        batch_size=1,
    ):
        assert tau in TAUS, f"tau must be one of {TAUS}"
        
        if proteins == None:
            proteins = list(FAST_FOLDING_PROTEINS.keys())
        else:
            for protein in proteins:
                assert protein in FAST_FOLDING_PROTEINS.keys(), f'{protein} is not a valid option'

        num_shards = len(list(filter(lambda x: 'shards-' in x, os.listdir(base))))

        self.base_path = base 
        self.proteins = proteins 
        self.tau = tau
        self.time_sort = True
        self.num_shards = num_shards
        self.batch_size = batch_size   

        self.atom_arrays = {
            protein: pdb.PDBFile.read(
               os.path.join(base, protein + ".pdb")
            ).get_structure()[0] for protein in proteins
        }

        if padded: 
            self.pad = ProteinPad(
                pad_size=max([FAST_FOLDING_PROTEINS[protein] for protein in proteins]))
        else: 
            self.pad = lambda x: x



        def build_webdataset(sample):
            if self.tau == 0:
                key, coord1 = sample
                coords = [ coord1 ]
            else:
                key, coord1, coord2 = sample
                coords = [ coord1, coord2 ]
            protein = key.split('_')[-1]
            template = self.atom_arrays[protein]
            data = []
            for coord in coords:
                new_aa = deepcopy(template)
                new_aa.coord = coord
                data.append(
                    self.pad.transform(
                        ProteinDatum.from_atom_array(
                            new_aa,
                            header={'idcode': protein, 'resolution': None}
                        )
                    )
                )
            return data

        keys = ('__key__', 'coord.npy')
        if self.tau > 0:
            keys = keys + (f'coord_{self.tau}.npy', )

        self.web_ds = iter(
            wds.WebDataset(base + 'shards-' + '{00000..%05d}.tar' % (num_shards - 1))
            .decode()
            .to_tuple(*keys)
            .map(build_webdataset)
            .batched(batch_size, collation_fn=lambda x: x)
        )

        self.splits = { 'train': self }

    def __len__(self):
        return self.num_shards * 1000 // self.batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return next(self.web_ds)
    
    def __getitem__(self, index):
        return next(self)