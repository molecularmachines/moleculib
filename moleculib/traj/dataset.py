import os
from functools import partial
from typing import Optional, Dict, Any
from pathlib import Path
import pickle

import numpy as np

from moleculib.protein.datum import ProteinDatum
import os 
import numpy as np
import biotite.structure.io.pdb as pdb


from ..protein.datum import ProteinDatum
from ..protein.transform import ProteinCrop
from ..abstract.dataset import Dataset, PreProcessedDataset


class MODELDataset(Dataset):
    """
    Holds ProteinDatum dataset across trajectories
    as catalogued in Molecular Dynamics Extended Library (MODEL)

    Arguments:
    ----------
    base_path : str
        directory to store all PDB files
    format : str
        the file format for each PDB file, either "npz" or "pdb"
    """

    def __init__(
        self,
        base_path: str,
        frac: float = 1.0,
        transform: list = [],
        num_steps=5,
        crop_size: int = 64,
    ):
        super().__init__()
        self.base_path = Path(base_path) / "models"
        self.num_steps = num_steps
        self.transform = transform

        self.traj = [
            os.path.join(self.base_path, traj_dir)
            for traj_dir in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, traj_dir))
        ]
        np.random.shuffle(self.traj)
        self.traj = self.traj[: int(frac * len(self.traj))]
        models_per_traj = [sorted(os.listdir(traj_dir)) for traj_dir in self.traj]
        self.models_per_traj = [
            [os.path.join(traj_dir, model) for model in models]
            for traj_dir, models in zip(self.traj, models_per_traj)
        ]
        self.num_models_per_traj = [len(models) for models in self.models_per_traj]

        self.models_per_traj = [
            models
            for models, num_models in zip(
                self.models_per_traj, self.num_models_per_traj
            )
            if num_models > self.num_steps
        ]
        self.num_models_per_traj = [
            num_models
            for num_models in self.num_models_per_traj
            if num_models > self.num_steps
        ]

        self.protein_crop = ProteinCrop(crop_size=crop_size)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.models_per_traj)

    def __getitem__(self, idx):
        start = np.random.randint(0, self.num_models_per_traj[idx] - self.num_steps)

        data = []
        for i in range(self.num_steps):
            datum = ProteinDatum.from_filepath(self.models_per_traj[idx][start + i])
            data.append(datum)

        proxy = data[0]
        diff = proxy.residue_token.shape[0] - self.crop_size
        cut = np.random.randint(low=0, high=diff) if diff > 0 else None
        data = map(partial(self.protein_crop.transform, cut=cut), data)

        if self.transform is not None:
            for transformation in self.transform:
                data = map(transformation.transform, data)

        return list(data)


class AdKEquilibriumDataset(PreProcessedDataset):
    """
    Holds ProteinDatum dataset across trajectories
    as catalogued in Molecular Dynamics Extended Library (MODEL)

    Arguments:
    ----------
    base_path : str
        directory to store all PDB files
    format : str
        the file format for each PDB file, either "npz" or "pdb"
    """

    def __init__(
        self,
        base_path: str,
        transform: list = [],
        num_steps=15,
    ):
        base_path = os.path.join(base_path, 'AdKEquilibrium.pyd')
        with open(base_path, 'rb') as fin:
            print('Loading data...')
            splits = pickle.load(fin)
        self.num_steps = num_steps
        super().__init__(splits, transform, shuffle=False, pre_transform=False)


class AdKTransitionsDataset(Dataset):
    """
    Holds ProteinDatum dataset across trajectories
    as catalogued in Molecular Dynamics Extended Library (MODEL)

    Arguments:
    ----------
    base_path : str
        directory to store all PDB files
    format : str
        the file format for each PDB file, either "npz" or "pdb"
    """

    def __init__(
        self,
        base_path: str,
        frac: float = 1.0,
        transform: list = [],
        num_steps=15,
        split: str = "train",
    ):
        super().__init__()
        self.base_path = Path(base_path)
        self.num_steps = num_steps
        self.transform = transform
        self.crop_size = 214

        trajectory_folder = os.listdir(self.base_path)
        trajectories = []
        for folder in trajectory_folder:
            files = os.listdir(self.base_path / folder)
            files = [
                (int(file[file.index("_") + 1 : file.index(".")]), file)
                for file in files
                if file.endswith(".pdb")
            ]
            files.sort()
            files = [file[1] for file in files]
            trajectories.append([folder, files])
        self.trajectories = trajectories
        

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        # load all proteins from the trajectory and return the list
        data = []
        folder, traj = self.trajectories[idx]
        for model in traj:
            datum = ProteinDatum.from_filepath(self.base_path / folder / model, format='pdb')
            data.append(datum)
        return data



NUM_STEPS = 1_000

class AtlasDataset(Dataset):

    def __init__(
        self,
        base_path: str,
        tau: int = None,
        transform: list = [],
        mode: str = 'next_step',
        single_protein: bool = False,
        min_sequence_size: int = 12,
        max_sequence_size: int = 512,
        expansion_factor: int =  100,
    ):  
        self.base_path = base_path
        self.num_steps = tau
        self.single_protein = single_protein
        
        pdbids = []
        with open(f'{self.base_path}/sequence_sizes.csv', 'r') as f:
            for line in f:
                if 'pdbid' in line:
                    continue
                pdbid, sequence_size = line.strip().split(',')
                if int(sequence_size) < min_sequence_size or int(sequence_size) > max_sequence_size:
                    continue
                pdbids.append(pdbid)

        self.pdbids = pdbids
        print(f'Detected {len(self.pdbids)} complete trajectories')

        if single_protein:
            self.pdbids = [self.pdbids[0]] * 2048 
        
        if expansion_factor > 0:  
            self.pdbids = self.pdbids * expansion_factor

        self.splits = { 'train': self }
        super().__init__(transform=transform)

        self.mode = mode
            
    def __len__(self):
        return len(self.pdbids)
    
    def _getitem(self, i):
        if self.num_steps != 0:
            pdbid = self.pdbids[i]
            traj_sample = np.random.randint(1, 4)
            path = '{}/{}/{}'.format(self.base_path, pdbid, traj_sample)
            t = np.random.randint(0, NUM_STEPS - self.num_steps)
            path1 = '{}/{}.bcif'.format(path, t)
            path2 = '{}/{}.bcif'.format(path, t + self.num_steps)
            protein_datum_i = ProteinDatum.from_filepath(path1)
            protein_datum_j = ProteinDatum.from_filepath(path2)
            return [protein_datum_i, protein_datum_j] 
        elif self.num_steps == 0:
            pdbid = self.pdbids[i]
            traj_sample = np.random.randint(1, 4)
            path = '{}/{}/{}'.format(self.base_path, pdbid, traj_sample)
            t = np.random.randint(0, NUM_STEPS - 1)
            path1 = '{}/{}.bcif'.format(path, t)
            protein_datum_i = ProteinDatum.from_filepath(path1)
            return [protein_datum_i]
    
    def to_dict(self, attrs=None):
        if attrs is None:
            attrs = vars(self).keys()
        dict_ = {}
        for attr in attrs:
            obj = getattr(self, attr)
            # strings are not JAX types
            if type(obj) == str:
                continue
            if type(obj) in [list, tuple]:
                if type(obj[0]) not in [int, float]:
                    continue
                obj = np.array(obj)
            dict_[attr] = obj
        return dict_



class AtlasEIF4EDataset(PreProcessedDataset):

    def __init__(self, base_path, transform=[]):
        base_path = os.path.join(base_path, "ATLAS4E1.pyd")
        with open(base_path, "rb") as fin:
            print("Loading data...")
            splits = pickle.load(fin)
        super().__init__(splits, transform, shuffle=False)

    def to_dict(self, attrs=None):
        if attrs is None:
            attrs = vars(self).keys()
        dict_ = {}
        for attr in attrs:
            obj = getattr(self, attr)
            # strings are not JAX types
            if type(obj) == str:
                continue
            if type(obj) in [list, tuple]:
                if type(obj[0]) not in [int, float]:
                    continue
                obj = np.array(obj)
            dict_[attr] = obj
        return dict_


import logging
from typing import Optional
import biotite.structure.io.pdb as pdb


class TimewarpDataset:
    """Exposes datasets from the Timewarp paper."""

    def __init__(
        self,
        dataset: str,
        split: str,
        tau: int,
        max_files: Optional[int] = None,
    ):
        base = "/mas/projects/molecularmachines/db/timewarp2/"
        self.base_path = os.path.join(base, dataset, split)
        self.counter = 0
        self.tau = tau

        self.files = self._list_files()
        if len(self.files) == 0:
            raise ValueError(f"No files found in {self.base_path}")

        print(f"Found {len(self.files)} files in {self.base_path}")
        if max_files is not None:
            self.files = self.files[:max_files]
            print(f"Using {max_files} file(s)...")

        print(f"Loading first file: {self.files[0]}")
        self._load_coords(self.files[0])

    def _list_files(self):
        files_with_extension = set()
        for filename in os.listdir(self.base_path):
            if filename.endswith(".npz") and not filename.startswith("."):
                files_with_extension.add(os.path.join(self.base_path, filename))
        return sorted(list(files_with_extension))

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
        
        if self.tau == 0:
            return p1
        
        self.atom_array._coord = self.coords[idxx + self.tau]
        p2 = ProteinDatum.from_atom_array(
            self.atom_array,
            header=dict(
                idcode=None,
                resolution=None,
            ),
        )

        return [p2, p1]
    

class TimewarpDatasetPreprocessed(PreProcessedDataset):

    def __init__(self, split_info: Dict[str, Any]):
        self.splits = {}
        for split in split_info:
            self.splits[split] = TimewarpDataset(
                **split_info[split]
            )
        

