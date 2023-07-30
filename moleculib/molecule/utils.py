import os
from pathlib import Path

from biotite.structure import (
    filter_amino_acids,
    filter_monoatomic_ions,
    filter_nucleotides,
)

import numpy as np
import biotite.structure.io.mmtf as mmtf

home_dir = str(Path.home())
config = {"cache_dir": os.path.join(home_dir, ".cache", "moleculib")}

_solvent_list = ["DOD", "HOH", "SOL"]  # added DOD
_unknown_list = ["UNX", "UNL"]  # unknown atom or ion, or ligand


def pdb_to_atom_array(mmtf_file):
    atom_array = mmtf.get_structure(
        mmtf_file,
        model=1,
        extra_fields=["atom_id", "b_factor", "occupancy", "charge"],
        include_bonds=True,
    )
    # keep only molecular atoms
    atom_array = atom_array[~filter_amino_acids(atom_array)]
    atom_array = atom_array[~filter_nucleotides(atom_array)]
    atom_array = atom_array[~filter_monoatomic_ions(atom_array)]
    atom_array = atom_array[~np.isin(atom_array.res_name, _solvent_list)]
    atom_array = atom_array[
        ~np.isin(atom_array.res_name, _unknown_list)
    ]  # unknown atom or ion
    return atom_array


def pids_file_to_list(pids_path):
    with open(pids_path) as f:
        pids_str = f.read()
    return pids_str.rstrip().split(",")


def pad_array(array, total_size, bonds=False):
    shape = array.shape[1:]
    size = array.shape[0]
    diff = total_size - size
    assert diff >= 0
    if diff == 0:
        return array
    if bonds:
        # pad with -1 to match biotite BondType
        return np.pad(array, (0, diff), constant_values=-1)

    pad = np.zeros((diff, *shape), dtype=array.dtype)
    return np.concatenate((array, pad), axis=0)
