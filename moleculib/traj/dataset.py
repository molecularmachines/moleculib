import os
import pickle
import traceback
from functools import partial
from pathlib import Path
from tempfile import gettempdir
from typing import List, Union

import biotite
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map

from ..protein.datum import ProteinDatum
from ..protein.batch import ProteinCollator, PadBatch
from ..protein.transform import ProteinCrop


class MultiPadBatch(ProteinCollator):
    def __init__(self, pad_mask, **kwargs):
        super().__init__()
        self.pad_mask = pad_mask
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    @classmethod
    def collate(cls, stream):
        sides = list(zip(*stream))
        try:
            batches = [PadBatch.collate(side) for side in sides]
        except:
            breakpoint()

        return batches


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


class AdKEqDataset(Dataset):
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
        start = np.random.randint(1, self.num_models - self.num_steps)

        datum_prev = ProteinDatum.from_filepath(self.base_path / self.models[start - 1])
        datum = ProteinDatum.from_filepath(self.base_path / self.models[start])
        datum1 = ProteinDatum.from_filepath(
            self.base_path / self.models[start + self.num_steps]
        )

        data = [datum_prev, datum, datum1]
        if self.transform is not None:
            for transformation in self.transform:
                data = map(transformation.transform, data)

        data = list(data)
        data[1].atom_velocity = data[1].atom_coord - data[0].atom_coord

        return data[1:]

class TrajNextStepDataset(Dataset):
    """
    Traj Next Step Dataset returns a tuple of (ProteinDatum at time step t,ProteinDatum at time step t+delta_t).
    frame start and end will determine the timesteps used for training.

    Arguments:
    ----------
    pdb_id : str
        pdb id for the md traj
    traj_id : str
        subset select the trajectory
    traj_type : str
        select between "GPCR" or "MODEL"
    frame_start: int
        start frame
    frame_end: int
        end frame
        
    """

    def __init__(
        self,
        pdb_id:str,
        traj_type:str,
        frame_start: int,
        frame_end: int,
        frames_dir:str,
        traj_id:str=None,
        transform:list=None,
    ):
        super().__init__()
        self.pdb_id = pdb_id
        self.traj_type = traj_type
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.frames_dir = frames_dir
        self.traj_id = traj_id
        self.transform = transform
        
        if self.traj_type == "GPCR":
            assert self.traj_id != None
            self.pdbs_dir =pdb_id+"/"+self.traj_id 
        elif self.traj_type == "MODEL":
            self.pdbs_dir =pdb_id
            
        fulltraj_dir = os.path.join(self.frames_dir,self.pdbs_dir)
        self.traj_files = [os.path.join(fulltraj_dir,fname) for fname in os.listdir(fulltraj_dir)[self.frame_start:self.frame_end]]

    def __len__(self):
        return len(self.traj_files)-1

    def __getitem__(self, idx):
        datum_current = ProteinDatum.from_filepath(self.traj_files[idx])
        datum_next = ProteinDatum.from_filepath(self.traj_files[idx+1])

        data = [datum_current, datum_next]
        
        #Leaving this in - in case we want to apply any transformations to the datum later.
        if self.transform is not None:
            for transformation in self.transform:
                data = map(transformation.transform, data)

        return data