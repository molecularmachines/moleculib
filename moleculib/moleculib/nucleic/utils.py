import os
from pathlib import Path

from biotite.structure import filter_nucleotides
from biotite.structure.io.pdb import PDBFile


import numpy as np

home_dir = str(Path.home()) #not sure what this do
config = {"cache_dir": os.path.join(home_dir, ".cache", "moleculib")} #not sure either


def pdb_to_atom_array(pdb_path):
    pdb_file = PDBFile.read(pdb_path)
    atom_array = pdb_file.get_structure(
        model=1, extra_fields=["atom_id", "b_factor", "occupancy", "charge"])
    aa_filter = filter_nucleotides(atom_array)
    atom_array = atom_array[aa_filter]
    return atom_array

#not sure what it does-
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
