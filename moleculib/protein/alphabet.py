from collections import OrderedDict
from typing import List

import numpy as np
from ordered_set import OrderedSet

UNK_TOKEN = 1

# data collected by Jonathan King in
# SidechainNet https://github.com/jonathanking/sidechainnet
# Further processed by Eric Alcaide in
# MP-NeRF: Massively Parallel Natural Extension of Reference Frame
# https://github.com/EleutherAI/mp_nerf
# And modified in this version to allow cycles in sidechains
# number of listed properties significantly reduced from original
#
# This document inherits the license from SidechainNet
#
# Copyright 2020 Jonathan King
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

sidechain_chemistry_per_residue = OrderedDict(
    ALA=dict(
        bonds=[["CA", "CB"]],
        angles=[["N", "CA", "CB"], ["C", "CA", "CB"]],
        dihedrals=[["C", "N", "CA", "CB"]],
        flippable=[],
    ),
    ARG=dict(
        bonds=[
            ["CA", "CB"],
            ["CB", "CG"],
            ["CG", "CD"],
            ["CD", "NE"],
            ["NE", "CZ"],
            ["CZ", "NH1"],
            ["CZ", "NH2"],
        ],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD"],
            ["CG", "CD", "NE"],
            ["CD", "NE", "CZ"],
            ["NE", "CZ", "NH1"],
            ["NE", "CZ", "NH2"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "NE"],
            ["CG", "CD", "NE", "CZ"],
            ["CD", "NE", "CZ", "NH1"],
            ["CD", "NE", "CZ", "NH2"],
        ],
        flippable=[],
    ),
    ASN=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "OD1"], ["CG", "ND2"]],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "OD1"],
            ["CB", "CG", "ND2"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "OD1"],
            ["CA", "CB", "CG", "ND2"],
        ],
        flippable=[],
    ),
    ASP=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "OD1"], ["CG", "OD2"]],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "OD1"],
            ["CB", "CG", "OD2"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG"],
            # ["CA", "CB", "CG", "OD1"],
            # ["CA", "CB", "CG", "OD2"],
        ],
        flippable=[
            ["OD1", "OD2"],
        ],
    ),
    CYS=dict(
        bonds=[["CA", "CB"], ["CB", "SG"]],
        angles=[["N", "CA", "CB"], ["C", "CA", "CB"], ["CA", "CB", "SG"]],
        dihedrals=[["C", "N", "CA", "CB"], ["N", "CA", "CB", "SG"]],
        flippable=[],
    ),
    GLN=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "OE1"], ["CD", "NE2"]],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD"],
            ["CG", "CD", "OE1"],
            ["CG", "CD", "NE2"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "OE1"],
            ["CB", "CG", "CD", "NE2"],
        ],
        flippable=[],
    ),
    GLU=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "OE1"], ["CD", "OE2"]],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD"],
            ["CG", "CD", "OE1"],
            ["CG", "CD", "OE2"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "OE1"],
            ["CB", "CG", "CD", "OE2"],
        ],
        flippable=[["OE1", "OE2"]],
    ),
    GLY=dict(bonds=[], angles=[], dihedrals=[], flippable=[]),
    HIS=dict(
        bonds=[
            ["CA", "CB"],
            ["CB", "CG"],
            ["CG", "ND1"],
            ["ND1", "CE1"],
            ["CE1", "NE2"],
            ["NE2", "CD2"],
        ],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "ND1"],
            ["CG", "ND1", "CE1"],
            ["ND1", "CE1", "NE2"],
            ["CE1", "NE2", "CD2"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "ND1"],
            ["CB", "CG", "ND1", "CE1"],
            ["CG", "ND1", "CE1", "NE2"],
            ["ND1", "CE1", "NE2", "CD2"],
        ],
        flippable=[],
    ),
    ILE=dict(
        bonds=[["CA", "CB"], ["CB", "CG1"], ["CG1", "CD1"], ["CB", "CG2"]],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG1"],
            ["CB", "CG1", "CD1"],
            ["CA", "CB", "CG2"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG1"],
            ["CA", "CB", "CG1", "CD1"],
            ["N", "CA", "CB", "CG2"],
        ],
        flippable=[],
    ),
    LEU=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "CD1"], ["CG", "CD2"]],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD1"],
            ["CB", "CG", "CD2"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD1"],
            ["CA", "CB", "CG", "CD2"],
        ],
        flippable=[],
    ),
    LYS=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "CE"], ["CE", "NZ"]],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD"],
            ["CG", "CD", "CE"],
            ["CD", "CE", "NZ"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["CB", "CG", "CD", "CE"],
            ["CG", "CD", "CE", "NZ"],
        ],
        flippable=[],
    ),
    MET=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "SD"], ["SD", "CE"]],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "SD"],
            ["CG", "SD", "CE"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "SD"],
            ["CB", "CG", "SD", "CE"],
        ],
        flippable=[],
    ),
    PHE=dict(
        bonds=[
            ["CA", "CB"],
            ["CB", "CG"],
            ["CG", "CD1"],
            ["CD1", "CE1"],
            ["CE1", "CZ"],
            ["CZ", "CE2"],
            ["CE2", "CD2"],
        ],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD1"],
            ["CG", "CD1", "CE1"],
            ["CD1", "CE1", "CZ"],
            ["CE1", "CZ", "CE2"],
            ["CZ", "CE2", "CD2"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG"],
            # ["CA", "CB", "CG", "CD1"],
            ["CB", "CG", "CD1", "CE1"],
            ["CG", "CD1", "CE1", "CZ"],
            ["CD1", "CE1", "CZ", "CE2"],
            ["CE1", "CZ", "CE2", "CD2"],
        ],
        flippable=[
            ["CD1", "CD2"],
            ["CE1", "CE2"],
        ],
    ),
    PRO=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "CD"]],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
        ],
        flippable=[],
    ),
    SER=dict(
        bonds=[["CA", "CB"], ["CB", "OG"]],
        angles=[["N", "CA", "CB"], ["C", "CA", "CB"], ["CA", "CB", "OG"]],
        dihedrals=[["C", "N", "CA", "CB"], ["N", "CA", "CB", "OG"]],
        flippable=[],
    ),
    THR=dict(
        bonds=[["CA", "CB"], ["CB", "OG1"], ["CB", "CG2"]],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "OG1"],
            ["CA", "CB", "CG2"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "OG1"],
            ["N", "CA", "CB", "CG2"],
        ],
        flippable=[],
    ),
    TRP=dict(
        bonds=[
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
            ["CD2", "CE2"],
            ["CD2", "CG"],
        ],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD1"],
            ["CG", "CD1", "NE1"],
            ["CD1", "NE1", "CE2"],
            ["NE1", "CE2", "CZ2"],
            ["CE2", "CZ2", "CH2"],
            ["CZ2", "CH2", "CZ3"],
            ["CH2", "CZ3", "CE3"],
            ["CZ3", "CE3", "CD2"],
            ["CE3", "CD2", "CG"],
            ["CE3", "CD2", "CE2"],
            ["CD1", "CG", "CD2"],
            ["CD2", "CE2", "CZ2"],
            ["CZ2", "CE2", "NE1"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD1"],
            ["CB", "CG", "CD1", "NE1"],
            ["CG", "CD1", "NE1", "CE2"],
            ["CD1", "NE1", "CE2", "CZ2"],
            ["NE1", "CE2", "CZ2", "CH2"],
            ["CE2", "CZ2", "CH2", "CZ3"],
            ["CZ2", "CH2", "CZ3", "CE3"],
            ["CH2", "CZ3", "CE3", "CD2"],
            ["CE3", "CD2", "CE2", "CZ2"],
            ["CE3", "CD2", "CE2", "NE1"],
            ["CG", "CD2", "CE2", "CZ2"],
            ["CG", "CD2", "CE2", "NE1"],
            ["CZ3", "CE3", "CD2", "CG"],
            ["CZ3", "CE3", "CD2", "CE2"],
            ["CE3", "CD2", "CG", "CB"],
            ["CE2", "CD2", "CG", "CD1"],
            ["CD1", "NE1", "CE2", "CD2"],
        ],
        flippable=[],
    ),
    TYR=dict(
        bonds=[
            ["CA", "CB"],
            ["CB", "CG"],
            ["CG", "CD1"],
            ["CD1", "CE1"],
            ["CE1", "CZ"],
            ["CZ", "OH"],
            ["CZ", "CE2"],
            ["CE2", "CD2"],
            ["CD2", "CG"],
        ],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["CB", "CG", "CD1"],
            ["CG", "CD1", "CE1"],
            ["CD1", "CE1", "CZ"],
            ["CE1", "CZ", "OH"],
            ["CE1", "CZ", "CE2"],
            ["CZ", "CE2", "CD2"],
            ["CD1", "CG", "CD2"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG"],
            # ["CA", "CB", "CG", "CD1"],
            ["CB", "CG", "CD1", "CE1"],
            ["CG", "CD1", "CE1", "CZ"],
            ["CD1", "CE1", "CZ", "OH"],
            ["CD1", "CE1", "CZ", "CE2"],
            ["CE1", "CZ", "CE2", "CD2"],
            ["CZ", "CE2", "CD2", "CG"],
            ["CE2", "CD2", "CG", "CD1"],
            # ["CA", "CB", "CG", "CD2"],
            ["CD2", "CG", "CD1", "CE1"],
        ],
        flippable=[
            ["CE1", "CE2"],
            ["CD1", "CD2"],
        ],
    ),
    VAL=dict(
        bonds=[["CA", "CB"], ["CB", "CG1"], ["CB", "CG2"]],
        angles=[
            ["N", "CA", "CB"],
            ["C", "CA", "CB"],
            ["CA", "CB", "CG1"],
            ["CA", "CB", "CG2"],
        ],
        dihedrals=[
            ["C", "N", "CA", "CB"],
            ["N", "CA", "CB", "CG1"],
            ["N", "CA", "CB", "CG2"],
        ],
        flippable=[],
    ),
)

# build base vocabularies for atoms
backbone_atoms = ["N", "CA", "C", "O"]
backbone_chemistry = dict(
    bonds=[["N", "CA"], ["CA", "C"], ["C", "O"]],
    angles=[["CA", "C", "O"], ["N", "CA", "C"]],
    dihedrals=[["N", "CA", "C", "O"]],
    flippable=[],
)
special_tokens = ["PAD", "UNK"]

atoms_per_residue = {
    res: backbone_atoms + sidechain_atoms
    for (res, sidechain_atoms) in sidechain_atoms_per_residue.items()
}
atoms_per_residue['PAD'] = []
atoms_per_residue['UNK'] = backbone_atoms 

all_atoms = list(OrderedSet(sum(list(atoms_per_residue.values()), [])))
all_atoms =  all_atoms
all_atoms_tokens = np.arange(len(all_atoms))

all_residues = list(sidechain_atoms_per_residue.keys())
all_residues = special_tokens + all_residues
all_residues_tokens = np.arange(len(all_residues))

# and base vocabularies for chemistry
def geometry_to_per_residue_indexing_array(property="bonds"):
    connecting_atoms_per_residue = OrderedDict()
    for token in special_tokens:
        connecting_atoms_per_residue[token] = []
    for (res, sidechain_indices) in sidechain_chemistry_per_residue.items():
        connecting_atoms_per_residue[res] = (
            backbone_chemistry[property] + sidechain_indices[property]
        )

    index_per_residue = OrderedDict()
    for (res, residue_indices) in connecting_atoms_per_residue.items():
        index_per_residue[res] = [
            [atoms_per_residue[res].index(i) for i in atom_indices]
            for atom_indices in residue_indices
        ]

    num_indices = len(backbone_chemistry[property][0]) if property != "flippable" else 2
    indices_arr_len = max([len(indices) for indices in index_per_residue.values()])
    indices_arr = np.zeros((len(all_residues), indices_arr_len, num_indices))
    indices_mask = np.zeros((len(all_residues), indices_arr_len, 1)).astype(np.bool_)

    for idx, indices in enumerate(index_per_residue.values()):
        len_difference = indices_arr_len - len(indices)
        indices = np.array(indices)
        indices_mask[idx] = True
        if len_difference != 0:
            pad = np.array([[indices_arr_len] * num_indices] * len_difference)
            indices = (
                np.concatenate((np.array(indices), pad), axis=0)
                if len(indices) > 0
                else pad
            )
            indices_mask[idx, -len_difference:] = False
        indices_arr[idx] = indices

    return indices_arr, indices_mask


bonds_arr, bonds_mask = geometry_to_per_residue_indexing_array("bonds")
angles_arr, angles_mask = geometry_to_per_residue_indexing_array("angles")
dihedrals_arr, dihedrals_mask = geometry_to_per_residue_indexing_array("dihedrals")
flippable_arr, flippable_mask = geometry_to_per_residue_indexing_array("flippable")


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
