import os
from functools import partial
from pathlib import Path

import numpy as np

from moleculib.protein.datum import ProteinDatum
import os 
import numpy as np

from ..protein.datum import ProteinDatum
from ..protein.transform import ProteinCrop
from ..abstract.dataset import Dataset


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


class AdKEquilibriumDataset(Dataset):
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

        files = os.listdir(self.base_path)
        files = [
            (int(file[file.index("_") + 1 : file.index(".")]), file)
            for file in files
            if file.endswith(".pdb")
        ]
        files.sort()
        files = [file[1] for file in files]

        self.models = files
        if split == "train":
            self.models = self.models[: int(0.6 * len(self.models))]
        elif split == "val":
            self.models = self.models[
                int(0.6 * len(self.models)) : int(0.8 * len(self.models))
            ]
        elif split == "test":
            self.models = self.models[int(0.8 * len(self.models)) :]

        self.protein_crop = ProteinCrop(crop_size=self.crop_size)
        self.num_models = len(self.models)

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        datum = ProteinDatum.from_filepath(self.base_path / self.models[idx], format='pdb')
        return [datum]


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
        num_steps: int = None,
        transform: list = [],
        mode: str = 'next_step',
        single_protein: bool = False,
        min_sequence_size: int = 12,
        max_sequence_size: int = 512,
        expansion_factor: int =  100,
    ):  
        self.base_path = base_path
        self.num_steps = num_steps
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
        if self.mode == 'next_step':
            pdbid = self.pdbids[i]
            traj_sample = np.random.randint(1, 4)
            path = '{}/{}/{}'.format(self.base_path, pdbid, traj_sample)
            t = np.random.randint(0, NUM_STEPS - self.num_steps)
            path1 = '{}/{}.bcif'.format(path, t)
            path2 = '{}/{}.bcif'.format(path, t + self.num_steps)
            protein_datum_i = ProteinDatum.from_filepath(path1)
            protein_datum_j = ProteinDatum.from_filepath(path2)
            return [protein_datum_i, protein_datum_j] 
        elif self.mode == 'single':
            pdbid = self.pdbids[i]
            traj_sample = np.random.randint(1, 4)
            path = '{}/{}/{}'.format(self.base_path, pdbid, traj_sample)
            t = np.random.randint(0, NUM_STEPS - 1)
            path1 = '{}/{}.bcif'.format(path, t)
            protein_datum_i = ProteinDatum.from_filepath(path1)
            return protein_datum_i