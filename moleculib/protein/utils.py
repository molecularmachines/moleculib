import os
from biotite.structure.io.pdb import PDBFile
from biotite.structure import filter_amino_acids
from pathlib import Path


home_dir = str(Path.home())
config = {
    'cache_dir': os.path.join(home_dir, '.cache', 'moleculib')
}


def pdb_to_atom_array(pdb_path):
    pdb_file = PDBFile.read(pdb_path)
    atom_array = pdb_file.get_structure(
        extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
    )
    aa_filter = filter_amino_acids(atom_array)
    atom_array = atom_array[:, aa_filter]
    return atom_array


def pids_file_to_list(pids_path):
    with open(pids_path) as f:
        pids_str = f.read()
    return pids_str.rstrip().split(",")
