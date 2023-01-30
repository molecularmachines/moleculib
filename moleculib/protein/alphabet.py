from collections import OrderedDict
from typing import List

import numpy as np
from ordered_set import OrderedSet

UNK_TOKEN = 1

sidechain_atoms_per_residue = OrderedDict(
    ALA=["CB"],
    ARG=["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    ASN=["CB", "CG", "OD1", "ND2"],
    ASP=["CB", "CG", "OD1", "OD2"],
    CYS=["CB", "SG"],
    GLN=["CB", "CG", "CD", "OE1", "NE2"],
    GLU=["CB", "CG", "CD", "OE1", "OE2"],
    GLY=[],
    HIS=["CB", "CG", "ND1", "CE1", "NE2", "CD2"],
    ILE=["CB", "CG1", "CD1", "CG2"],
    LEU=["CB", "CG", "CD1", "CD2"],
    LYS=["CB", "CG", "CD", "CE", "NZ"],
    MET=["CB", "CG", "SD", "CE"],
    PHE=["CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
    PRO=["CB", "CG", "CD"],
    SER=["CB", "OG"],
    THR=["CB", "OG1", "CG2"],
    TRP=["CB", "CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2"],
    TYR=["CB", "CG", "CD1", "CE1", "CZ", "OH", "CE2", "CD2"],
    VAL=["CB", "CG1", "CG2"],
)
sidechain_bonds_per_residue = OrderedDict(
    ALA=[["CA", "CB"]],
    ARG=[
        ["CA", "CB"],
        ["CB", "CG"],
        ["CG", "CD"],
        ["CD", "NE"],
        ["NE", "CZ"],
        ["CZ", "NH1"],
        ["CZ", "NH2"],
    ],
    ASN=[["CA", "CB"], ["CB", "CG"], ["CG", "OD1"], ["CG", "ND2"]],
    ASP=[["CA", "CB"], ["CB", "CG"], ["CG", "OD1"], ["CG", "OD2"]],
    CYS=[["CA", "CB"], ["CB", "SG"]],
    GLN=[["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "OE1"], ["CD", "NE2"]],
    GLU=[["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "OE1"], ["CD", "OE2"]],
    GLY=[],
    HIS=[
        ["CA", "CB"],
        ["CB", "CG"],
        ["CG", "ND1"],
        ["ND1", "CE1"],
        ["CE1", "NE2"],
        ["NE2", "CD2"],
    ],
    ILE=[["CA", "CB"], ["CB", "CG1"], ["CG1", "CD1"], ["CB", "CG2"]],
    LEU=[["CA", "CB"], ["CB", "CG"], ["CG", "CD1"], ["CG", "CD2"]],
    LYS=[["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "CE"], ["CE", "NZ"]],
    MET=[["CA", "CB"], ["CB", "CG"], ["CG", "SD"], ["SD", "CE"]],
    PHE=[
        ["CA", "CB"],
        ["CB", "CG"],
        ["CG", "CD1"],
        ["CD1", "CE1"],
        ["CE1", "CZ"],
        ["CZ", "CE2"],
        ["CE2", "CD2"],
    ],
    PRO=[["CA", "CB"], ["CB", "CG"], ["CG", "CD"]],
    SER=[["CA", "CB"], ["CB", "OG"]],
    THR=[["CA", "CB"], ["CB", "OG1"], ["CB", "CG2"]],
    TRP=[
        ["CA", "CB"],
        ["CB", "CG"],
        ["CG", "CD1"],
        ["CD1", "NE1"],
        ["NE1", "CE2"],
        ["CE2", "CZ2"],
        ["CZ2", "CH2"],
        ["CH2", "CZ3"],
        ["CZ3", "CE3"],
        ["CE3", "CD2"],
    ],
    TYR=[
        ["CA", "CB"],
        ["CB", "CG"],
        ["CG", "CD1"],
        ["CD1", "CE1"],
        ["CE1", "CZ"],
        ["CZ", "OH"],
        ["CZ", "CE2"],
        ["CE2", "CD2"],
    ],
    VAL=[["CA", "CB"], ["CB", "CG1"], ["CB", "CG2"]],
)


# build base vocabularies for atoms
backbone_atoms = ["N", "CA", "C", "O"]
backbone_bonds = [["N", "CA"], ["CA", "C"], ["C", "N"], ["C", "O"]]
special_tokens = ["PAD", "UNK"]

atoms_per_residue = {
    res: backbone_atoms + sidechain_atoms
    for (res, sidechain_atoms) in sidechain_atoms_per_residue.items()
}
for token in special_tokens:
    atoms_per_residue[token] = []

all_atoms = list(OrderedSet(sum(list(atoms_per_residue.values()), [])))
all_atoms = special_tokens + all_atoms
all_atoms_tokens = np.arange(len(all_atoms))

all_residues = list(sidechain_atoms_per_residue.keys())
all_residues = special_tokens + all_residues
all_residues_tokens = np.arange(len(all_residues))

# and base vocabularies for bonds
bonds_per_residue = OrderedDict()
for token in special_tokens:
    bonds_per_residue[token] = []
for (res, sidechain_bonds) in sidechain_bonds_per_residue.items():
    bonds_per_residue[res] = backbone_bonds + sidechain_bonds

bonds_index_per_residue = OrderedDict()
for (res, bonds) in bonds_per_residue.items():
    bonds_index_per_residue[res] = [
        (atoms_per_residue[res].index(v), atoms_per_residue[res].index(u))
        for (v, u) in bonds
    ]

bonds_arr_len = max([len(bonds) for bonds in bonds_index_per_residue.values()])
bonds_arr = np.zeros((len(all_residues), bonds_arr_len, 2))
bonds_mask = np.zeros((len(all_residues), bonds_arr_len, 1)).astype(np.bool_)

for idx, bonds in enumerate(bonds_index_per_residue.values()):
    len_difference = bonds_arr_len - len(bonds)
    bonds = np.array(bonds)
    bonds_mask[idx] = True
    if len_difference != 0:
        pad = np.array([[bonds_arr_len, bonds_arr_len]] * len_difference)
        bonds = (
            np.concatenate((np.array(bonds), pad), axis=0) if len(bonds) > 0 else pad
        )
        bonds_mask[idx, -len_difference:] = False
    bonds_arr[idx] = bonds


def _atom_to_all_residues_index(atom):
    def _atom_to_residue_index(residue):
        residue_atoms = atoms_per_residue[residue]
        mask = atom in residue_atoms
        index = residue_atoms.index(atom) if mask else 0
        return index, mask

    indices, masks = zip(*list(map(_atom_to_residue_index, all_residues)))
    return np.array(indices), np.array(masks)


def _index(lst: List[str], item: str) -> int:
    try:
        index = lst.index(item)
    except ValueError:
        index = UNK_TOKEN  # UNK
    return index


def atom_index(atom: str) -> int:
    return _index(all_atoms, atom)


def get_residue_index(residue: str) -> int:
    return _index(all_residues, residue)


atom_to_residues_index, atom_to_residues_mask = zip(
    *list(map(_atom_to_all_residues_index, all_atoms))
)
atom_to_residues_index = np.array(atom_to_residues_index)
atom_to_residues_mask = np.array(atom_to_residues_mask)
