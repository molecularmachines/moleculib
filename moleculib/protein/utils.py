import os
from pathlib import Path

from biotite.structure import filter_amino_acids
from biotite.structure.io.pdb import PDBFile
from .datum import ProteinDatum
import numpy as np
from einops import rearrange

home_dir = str(Path.home())
config = {"cache_dir": os.path.join(home_dir, ".cache", "moleculib")}


def pdb_to_atom_array(pdb_path):
    pdb_file = PDBFile.read(pdb_path)
    atom_array = pdb_file.get_structure(
        model=1, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
    )
    aa_filter = filter_amino_acids(atom_array)
    atom_array = atom_array[aa_filter]
    return atom_array


def pids_file_to_list(pids_path):
    with open(pids_path) as f:
        pids_str = f.read()
    return pids_str.rstrip().split(",")


def pad_array(array, total_size):
    shape = array.shape[1:]
    size = len(array)
    diff = total_size - size
    assert diff >= 0
    if diff == 0:
        return array

    pad = np.zeros((diff, *shape), dtype=array.dtype)
    return np.concatenate((array, pad), axis=0)


def measure_rmsd(x_: ProteinDatum, y_: ProteinDatum, mode="all_atom", align=False):
    mask = x_.atom_mask * y_.atom_mask
    x = x_.atom_coord.copy()
    y = y_.atom_coord.copy()
    if mode == "CA":
        x = x[:, 1:2, :]
        y = y[:, 1:2, :]
        mask = mask[:, 1:2]
        if align:
            x -= x.sum(0) / x_.residue_mask.sum()
            y -= y.sum(0) / y_.residue_mask.sum()
            R = rigid_Kabsch_3D(
                np.squeeze(x * mask[..., None]), np.squeeze(y * mask[..., None])
            )
            x = (R @ np.squeeze(x).T).T[:, None, :]

    dists = np.square(x - y).sum(-1)
    dists = (dists * mask).sum((-1, -2)) / (mask.sum((-1, -2)) + 1e-6)
    rmsd = np.sqrt(dists + 1e-6)
    rmsd = rmsd * (mask.sum((-1, -2)) > 0).astype(rmsd.dtype)

    return rmsd


def rigid_Kabsch_3D(Q, P):
    # Q (num points x 3) is the one to be rotated to match P (num points x 3)
    # Q, P need to be centered
    assert Q.shape[-1] == 3
    assert len(Q.shape) == 2
    assert Q.shape == P.shape
    Q = Q.astype(np.float64)
    P = P.astype(np.float64)
    B = np.einsum("ji,jk->ik", Q, P)
    U, S, Vh = np.linalg.svd(B)
    R = np.matmul(Vh.T, U.T)

    d = np.sign(np.linalg.det(R))
    if d < 0:
        return np.matmul(
            np.matmul(
                Vh.T,
                np.diag(np.array([1, 1, d])),
            ),
            U.T,
        )
    return R
