import os
from pathlib import Path

from biotite.structure import filter_amino_acids, filter_nucleotides
from biotite.structure.io.pdb import PDBFile
from .alphabet import atom_per_base

import numpy as np

home_dir = str(Path.home())
config = {"cache_dir": os.path.join(home_dir, ".cache", "moleculib")}


def pdb_to_atom_array(pdb_path):
    pdb_file = PDBFile.read(pdb_path)
    atom_array = pdb_file.get_structure(
        model=1, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
    )
    keep_indices = filter_amino_acids(atom_array)
    atom_array = atom_array[keep_indices]
    return atom_array


def pdb_to_dna_array(pdb_path):
    MAX_DNA_CHAINS = 2
    pdb_file = PDBFile.read(pdb_path)
    atom_array = pdb_file.get_structure(
        model=1, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
    )

    # filter nucleotides
    keep_indices = filter_nucleotides(atom_array)
    atom_array = atom_array[keep_indices]

    # filter max chains
    curr_num_chains = 0
    atom_idx = 0
    curr_chain_id = atom_array[0].chain_id
    while curr_num_chains < MAX_DNA_CHAINS:
        # last atom is the last of a chain
        if atom_idx == len(atom_array):
            break
        if atom_array[atom_idx].chain_id != curr_chain_id:
            curr_chain_id = atom_array[atom_idx].chain_id
            curr_num_chains += 1
        atom_idx += 1
    atom_array = atom_array[:atom_idx]

    # filter only ACGT
    dna_atom_types = ["DA", "DC", "DG", "DT"]
    keep_indices = [i for i, atom in enumerate(atom_array) if atom.res_name in dna_atom_types]
    if len(keep_indices) == 0:
        return []
    atom_array = atom_array[keep_indices]

    # retrieve centroid positions
    curr_res = atom_array[0].res_id
    centroids = []
    res_atoms = []
    num_res = 1

    for a in atom_array:
        if a.res_id != curr_res:
            # calcualte centroid for nucleotide
            res_atoms = np.stack(res_atoms, axis=0)
            centroid = np.mean(res_atoms, axis=0)
            centroids.append(centroid)

            # reset for next nucleotide
            res_atoms = []
            num_res += 1
            curr_res = a.res_id

        res_atoms.append(a.coord)

    # deal with last nucleotide
    if len(res_atoms):
        res_atoms = np.stack(res_atoms, axis=0)
        centroid = np.mean(res_atoms, axis=0)
        centroids.append(centroid)

    assert num_res == len(centroids)
    centroids = np.stack(centroids, axis=0)

    def is_base_atom(atom_name):
        if atom_name[-1] == "'":
            return False
        elif atom_name.startswith("OP"):
            # exclude phosphate oxygens
            return False
        elif atom_name.startswith("P"):
            # exclude phosphorus
            return False
        elif atom_name.startswith("H"):
            # exclude hydrogens
            return False
        return True

    # first pass for efficiency clear phosphate backbone atoms
    keep_indices = [i for i, atom in enumerate(atom_array) if is_base_atom(atom.atom_name)]
    if len(keep_indices) == 0:
        return []
    atom_array = atom_array[keep_indices]

    def atom_sorter(atom):
        return atom_per_base[atom.res_name[-1]].index(atom.atom_name)

    # sort atoms residue-wise
    curr_res_id = atom_array[0].res_id
    curr_atoms = []
    sorted_indices = []
    carry = 0

    for i, atom in enumerate(atom_array):
        if atom.res_id != curr_res_id:
            # sort atoms of same res
            sorted_atoms = sorted(enumerate(curr_atoms), key=lambda x: atom_sorter(x[1]))
            curr_order = [carry + s[0] for s in sorted_atoms]
            sorted_indices += curr_order
            # reset
            carry += len(curr_atoms)
            curr_res_id = atom.res_id
            curr_atoms = []
        curr_atoms.append(atom)

    # add last res
    if len(curr_atoms):
        sorted_atoms = sorted(enumerate(curr_atoms), key=lambda x: atom_sorter(x[1]))
        curr_order = [carry + s[0] for s in sorted_atoms]
        sorted_indices += curr_order

    assert len(sorted_indices) == len(atom_array)
    return atom_array[sorted_indices], centroids


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
