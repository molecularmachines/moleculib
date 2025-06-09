from collections import OrderedDict
from typing import List

import numpy as np
from ordered_set import OrderedSet

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

van_der_walls_radii = OrderedDict(  # in angstroms
    C=1.7,
    H=1.2,
    O=1.52,
    N=1.55,
    P=1.8,
    S=1.8,
    F=1.47,
)


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
        flippable=[],
        angles=[["N", "CA", "CB"], ["C", "CA", "CB"]],
        dihedrals=[["O", "C", "CA", "CB"]],
        bond_lens=[1.52],
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
        flippable=[],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["C", "CA", "CB"],
            ["CB", "CG", "CD"],
            ["CG", "CD", "NE"],
            ["CD", "NE", "CZ"],
            ["NE", "CZ", "NH1"],
            ["NE", "CZ", "NH2"],
            ["NH1", "CZ", "NH2"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["C", "CA", "CB", "CG"],
            ["O", "C", "CA", "CB"],
            ["CB", "CG", "CD", "NE"],
            ["CG", "CD", "NE", "CZ"],
            ["CD", "NE", "CZ", "NH1"],
            ["CD", "NE", "CZ", "NH2"],
        ],
        bond_lens=[1.53, 1.52, 1.52, 1.46, 1.33, 1.33, 1.33],
    ),
    ASN=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "OD1"], ["CG", "ND2"]],
        flippable=[],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["C", "CA", "CB"],
            ["CB", "CG", "OD1"],
            ["CB", "CG", "ND2"],
            ["OD1", "CG", "ND2"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "OD1"],
            ["CA", "CB", "CG", "ND2"],
            ["C", "CA", "CB", "CG"],
            ["O", "C", "CA", "CB"],
        ],
        bond_lens=[1.53, 1.51, 1.23, 1.33],
    ),
    ASP=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "OD1"], ["CG", "OD2"]],
        flippable=[["OD1", "OD2"]],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["C", "CA", "CB"],
            ["CB", "CG", "OD1"],
            ["CB", "CG", "OD2"],
            ["OD1", "CG", "OD2"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "OD1"],
            ["CA", "CB", "CG", "OD2"],
            ["C", "CA", "CB", "CG"],
            ["O", "C", "CA", "CB"],
        ],
        bond_lens=[1.53, 1.52, 1.25, 1.25],
    ),
    CYS=dict(
        bonds=[["CA", "CB"], ["CB", "SG"]],
        flippable=[],
        angles=[["N", "CA", "CB"], ["CA", "CB", "SG"], ["C", "CA", "CB"]],
        dihedrals=[
            ["N", "CA", "CB", "SG"],
            ["C", "CA", "CB", "SG"],
            ["O", "C", "CA", "CB"],
        ],
        bond_lens=[1.53, 1.81],
    ),
    GLN=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "OE1"], ["CD", "NE2"]],
        flippable=[],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["C", "CA", "CB"],
            ["CB", "CG", "CD"],
            ["CG", "CD", "OE1"],
            ["CG", "CD", "NE2"],
            ["OE1", "CD", "NE2"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["C", "CA", "CB", "CG"],
            ["O", "C", "CA", "CB"],
            ["CB", "CG", "CD", "OE1"],
            ["CB", "CG", "CD", "NE2"],
        ],
        bond_lens=[1.53, 1.52, 1.52, 1.23, 1.33],
    ),
    GLU=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "OE1"], ["CD", "OE2"]],
        flippable=[["OE1", "OE2"]],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["C", "CA", "CB"],
            ["CB", "CG", "CD"],
            ["CG", "CD", "OE1"],
            ["CG", "CD", "OE2"],
            ["OE1", "CD", "OE2"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["C", "CA", "CB", "CG"],
            ["O", "C", "CA", "CB"],
            ["CB", "CG", "CD", "OE1"],
            ["CB", "CG", "CD", "OE2"],
        ],
        bond_lens=[1.53, 1.52, 1.52, 1.25, 1.25],
    ),
    GLY=dict(
        bonds=[],
        flippable=[],
        angles=[],
        dihedrals=[],
        bond_lens=[],
    ),
    HIS=dict(
        bonds=[
            ["CA", "CB"],
            ["CB", "CG"],
            ["CG", "ND1"],
            ["ND1", "CE1"],
            ["CE1", "NE2"],
            ["NE2", "CD2"],
        ],
        flippable=[],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["C", "CA", "CB"],
            ["CB", "CG", "ND1"],
            ["CG", "ND1", "CE1"],
            ["ND1", "CE1", "NE2"],
            ["CE1", "NE2", "CD2"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "ND1"],
            ["C", "CA", "CB", "CG"],
            ["O", "C", "CA", "CB"],
            ["CB", "CG", "ND1", "CE1"],
            ["CG", "ND1", "CE1", "NE2"],
            ["ND1", "CE1", "NE2", "CD2"],
        ],
        bond_lens=[1.53, 1.49, 1.38, 1.32, 1.32, 1.37],
    ),
    ILE=dict(
        bonds=[["CA", "CB"], ["CB", "CG1"], ["CG1", "CD1"], ["CB", "CG2"]],
        flippable=[],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG1"],
            ["CA", "CB", "CG2"],
            ["C", "CA", "CB"],
            ["CB", "CG1", "CD1"],
            ["CG1", "CB", "CG2"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG1"],
            ["N", "CA", "CB", "CG2"],
            ["CA", "CB", "CG1", "CD1"],
            ["C", "CA", "CB", "CG1"],
            ["C", "CA", "CB", "CG2"],
            ["O", "C", "CA", "CB"],
            ["CD1", "CG1", "CB", "CG2"],
        ],
        bond_lens=[1.54, 1.53, 1.52, 1.53],
    ),
    LEU=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "CD1"], ["CG", "CD2"]],
        flippable=[],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["C", "CA", "CB"],
            ["CB", "CG", "CD1"],
            ["CB", "CG", "CD2"],
            ["CD1", "CG", "CD2"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD1"],
            ["CA", "CB", "CG", "CD2"],
            ["C", "CA", "CB", "CG"],
            ["O", "C", "CA", "CB"],
        ],
        bond_lens=[1.53, 1.53, 1.52, 1.52],
    ),
    LYS=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "CD"], ["CD", "CE"], ["CE", "NZ"]],
        flippable=[],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["C", "CA", "CB"],
            ["CB", "CG", "CD"],
            ["CG", "CD", "CE"],
            ["CD", "CE", "NZ"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["C", "CA", "CB", "CG"],
            ["O", "C", "CA", "CB"],
            ["CB", "CG", "CD", "CE"],
            ["CG", "CD", "CE", "NZ"],
        ],
        bond_lens=[1.53, 1.52, 1.52, 1.52, 1.49],
    ),
    MET=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "SD"], ["SD", "CE"]],
        flippable=[],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["C", "CA", "CB"],
            ["CB", "CG", "SD"],
            ["CG", "SD", "CE"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "SD"],
            ["C", "CA", "CB", "CG"],
            ["O", "C", "CA", "CB"],
            ["CB", "CG", "SD", "CE"],
        ],
        bond_lens=[1.53, 1.52, 1.80, 1.79],
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
            ["CD2", "CG"],
        ],
        flippable=[["CD1", "CD2"], ["CE1", "CE2"]],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["C", "CA", "CB"],
            ["CB", "CG", "CD1"],
            ["CB", "CG", "CD2"],
            ["CG", "CD1", "CE1"],
            ["CG", "CD2", "CE2"],
            ["CD1", "CG", "CD2"],
            ["CD1", "CE1", "CZ"],
            ["CE1", "CZ", "CE2"],
            ["CZ", "CE2", "CD2"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD1"],
            ["CA", "CB", "CG", "CD2"],
            ["C", "CA", "CB", "CG"],
            ["O", "C", "CA", "CB"],
            ["CB", "CG", "CD1", "CE1"],
            ["CB", "CG", "CD2", "CE2"],
            ["CG", "CD1", "CE1", "CZ"],
            ["CG", "CD2", "CE2", "CZ"],
            ["CD1", "CG", "CD2", "CE2"],
            ["CD1", "CE1", "CZ", "CE2"],
            ["CE1", "CD1", "CG", "CD2"],
            ["CE1", "CZ", "CE2", "CD2"],
        ],
        bond_lens=[1.53, 1.50, 1.39, 1.39, 1.38, 1.38, 1.39, 1.39],
    ),
    PRO=dict(
        bonds=[["CA", "CB"], ["CB", "CG"], ["CG", "CD"]],
        flippable=[],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["C", "CA", "CB"],
            ["CB", "CG", "CD"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD"],
            ["C", "CA", "CB", "CG"],
            ["O", "C", "CA", "CB"],
        ],
        bond_lens=[1.53, 1.49, 1.51],
    ),
    SER=dict(
        bonds=[["CA", "CB"], ["CB", "OG"]],
        flippable=[],
        angles=[["N", "CA", "CB"], ["CA", "CB", "OG"], ["C", "CA", "CB"]],
        dihedrals=[
            ["N", "CA", "CB", "OG"],
            ["C", "CA", "CB", "OG"],
            ["O", "C", "CA", "CB"],
        ],
        bond_lens=[1.53, 1.42],
    ),
    THR=dict(
        bonds=[["CA", "CB"], ["CB", "OG1"], ["CB", "CG2"]],
        flippable=[],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "OG1"],
            ["CA", "CB", "CG2"],
            ["C", "CA", "CB"],
            ["OG1", "CB", "CG2"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "OG1"],
            ["N", "CA", "CB", "CG2"],
            ["C", "CA", "CB", "OG1"],
            ["C", "CA", "CB", "CG2"],
            ["O", "C", "CA", "CB"],
        ],
        bond_lens=[1.54, 1.43, 1.52],
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
        flippable=[],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["C", "CA", "CB"],
            ["CB", "CG", "CD1"],
            ["CB", "CG", "CD2"],
            ["CG", "CD1", "NE1"],
            ["CG", "CD2", "CE3"],
            ["CG", "CD2", "CE2"],
            ["CD1", "CG", "CD2"],
            ["CD1", "NE1", "CE2"],
            ["NE1", "CE2", "CZ2"],
            ["NE1", "CE2", "CD2"],
            ["CE2", "CZ2", "CH2"],
            ["CE2", "CD2", "CE3"],
            ["CZ2", "CE2", "CD2"],
            ["CZ2", "CH2", "CZ3"],
            ["CH2", "CZ3", "CE3"],
            ["CZ3", "CE3", "CD2"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD1"],
            ["CA", "CB", "CG", "CD2"],
            ["C", "CA", "CB", "CG"],
            ["O", "C", "CA", "CB"],
            ["CB", "CG", "CD1", "NE1"],
            ["CB", "CG", "CD2", "CE3"],
            ["CB", "CG", "CD2", "CE2"],
            ["CG", "CD1", "NE1", "CE2"],
            ["CG", "CD2", "CE3", "CZ3"],
            ["CG", "CD2", "CE2", "NE1"],
            ["CG", "CD2", "CE2", "CZ2"],
            ["CD1", "CG", "CD2", "CE3"],
            ["CD1", "CG", "CD2", "CE2"],
            ["CD1", "NE1", "CE2", "CZ2"],
            ["CD1", "NE1", "CE2", "CD2"],
            ["NE1", "CD1", "CG", "CD2"],
            ["NE1", "CE2", "CZ2", "CH2"],
            ["NE1", "CE2", "CD2", "CE3"],
            ["CE2", "CZ2", "CH2", "CZ3"],
            ["CE2", "CD2", "CE3", "CZ3"],
            ["CZ2", "CE2", "CD2", "CE3"],
            ["CZ2", "CH2", "CZ3", "CE3"],
            ["CH2", "CZ2", "CE2", "CD2"],
            ["CH2", "CZ3", "CE3", "CD2"],
        ],
        bond_lens=[
            1.53,
            1.50,
            1.37,
            1.37,
            1.37,
            1.40,
            1.37,
            1.40,
            1.39,
            1.40,
            1.41,
            1.43,
        ],
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
        flippable=[["CE1", "CE2"], ["CD1", "CD2"]],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG"],
            ["C", "CA", "CB"],
            ["CB", "CG", "CD1"],
            ["CB", "CG", "CD2"],
            ["CG", "CD1", "CE1"],
            ["CG", "CD2", "CE2"],
            ["CD1", "CG", "CD2"],
            ["CD1", "CE1", "CZ"],
            ["CE1", "CZ", "OH"],
            ["CE1", "CZ", "CE2"],
            ["CZ", "CE2", "CD2"],
            ["OH", "CZ", "CE2"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG"],
            ["CA", "CB", "CG", "CD1"],
            ["CA", "CB", "CG", "CD2"],
            ["C", "CA", "CB", "CG"],
            ["O", "C", "CA", "CB"],
            ["CB", "CG", "CD1", "CE1"],
            ["CB", "CG", "CD2", "CE2"],
            ["CG", "CD1", "CE1", "CZ"],
            ["CG", "CD2", "CE2", "CZ"],
            ["CD1", "CG", "CD2", "CE2"],
            ["CD1", "CE1", "CZ", "OH"],
            ["CD1", "CE1", "CZ", "CE2"],
            ["CE1", "CD1", "CG", "CD2"],
            ["CE1", "CZ", "CE2", "CD2"],
            ["OH", "CZ", "CE2", "CD2"],
        ],
        bond_lens=[1.53, 1.51, 1.39, 1.39, 1.38, 1.38, 1.38, 1.39, 1.39],
    ),
    VAL=dict(
        bonds=[["CA", "CB"], ["CB", "CG1"], ["CB", "CG2"]],
        flippable=[],
        angles=[
            ["N", "CA", "CB"],
            ["CA", "CB", "CG1"],
            ["CA", "CB", "CG2"],
            ["C", "CA", "CB"],
            ["CG1", "CB", "CG2"],
        ],
        dihedrals=[
            ["N", "CA", "CB", "CG1"],
            ["N", "CA", "CB", "CG2"],
            ["C", "CA", "CB", "CG1"],
            ["C", "CA", "CB", "CG2"],
            ["O", "C", "CA", "CB"],
        ],
        bond_lens=[1.54, 1.53, 1.52],
    ),
)

# build base vocabularies for atoms
backbone_atoms = ["N", "CA", "C", "O"]
backbone_chemistry = dict(
    bonds=[["N", "CA"], ["CA", "C"], ["C", "O"]],
    angles=[["CA", "C", "O"], ["N", "CA", "C"]],
    dihedrals=[["N", "CA", "C", "O"]],
    flippable=[],
    bond_lens=[1.46, 1.52, 1.23],
)
special_tokens = ["PAD", "MASK", "UNK"]

atoms_per_residue = OrderedDict()
atoms_per_residue["PAD"] = []
atoms_per_residue["MASK"] = []
atoms_per_residue["UNK"] = backbone_atoms
for res, sidechain_atoms in sidechain_atoms_per_residue.items():
    atoms_per_residue[res] = backbone_atoms + sidechain_atoms

all_atoms = list(OrderedSet(sum(list(atoms_per_residue.values()), [])))
all_atoms = special_tokens + all_atoms
all_atoms_tokens = np.arange(len(all_atoms))

elements = list(OrderedSet([atom[0] for atom in all_atoms]))
all_atoms_elements = np.array([elements.index(atom[0]) for atom in all_atoms])
all_atoms_radii = np.array(
    [
        (van_der_walls_radii[atom[0]] if (atom[0] in van_der_walls_radii) else 0.0)
        for atom in all_atoms
    ]
)

all_residues = list(sidechain_atoms_per_residue.keys())
all_residues = special_tokens + all_residues
all_residues_tokens = np.arange(len(all_residues))
all_residues_atom_mask = np.array(
    [
        ([1] * len(atoms) + [0] * (14 - len(atoms)))
        for (_, atoms) in atoms_per_residue.items()
    ]
).astype(np.bool_)
all_residues_atom_tokens = np.array(
    [
        ([all_atoms.index(atom) for atom in atoms] + [0] * (14 - len(atoms)))
        for (_, atoms) in atoms_per_residue.items()
    ]
)


# and base vocabularies for chemistry
def geometry_to_per_residue_indexing_array(property="bonds"):
    connecting_atoms_per_residue = OrderedDict()
    for token in special_tokens:
        connecting_atoms_per_residue[token] = []
    for res, sidechain_indices in sidechain_chemistry_per_residue.items():
        connecting_atoms_per_residue[res] = (
            backbone_chemistry[property] + sidechain_indices[property]
        )

    index_per_residue = OrderedDict()
    for res, residue_indices in connecting_atoms_per_residue.items():
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

bond_lens_arr = [[] for _ in special_tokens]
bond_lens_arr += [
    backbone_chemistry["bond_lens"] + sidechain_chemistry["bond_lens"]
    for sidechain_chemistry in sidechain_chemistry_per_residue.values()
]
max_arr_size = max([len(v) for v in bond_lens_arr])
bond_lens_arr = np.array(
    [np.pad(arr, (0, max_arr_size - len(arr)), "constant") for arr in bond_lens_arr]
)

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


UNK_TOKEN = all_residues.index("UNK")


def _index(lst: List[str], item: str) -> int:
    try:
        index = lst.index(item)
    except ValueError:
        index = UNK_TOKEN  # UNK
    return index


def atom_index(atom: str) -> int:
    return _index(all_atoms, atom)


def get_residue_index(residue: str) -> int:
    if residue == "HSD" or residue == "HSE" or residue == "HSP" or residue == "HIE":
        residue = "HIS"
    return _index(all_residues, residue)


atom_to_residues_index, atom_to_residues_mask = zip(
    *list(map(_atom_to_all_residues_index, all_atoms))
)
atom_to_residues_index = np.array(atom_to_residues_index)
atom_to_residues_mask = np.array(atom_to_residues_mask)


HELIX_BACKBONE = """
ATOM      1  N   GLY A   1      -5.606  -2.251 -12.878  1.00  0.00           N
ATOM      2  CA  GLY A   1      -5.850  -1.194 -13.852  1.00  0.00           C
ATOM      3  C   GLY A   1      -5.186  -1.524 -15.184  1.00  0.00           C
ATOM      4  O   GLY A   1      -5.744  -1.260 -16.249  1.00  0.00           O
ATOM      6  N   GLY A   2      -3.991  -2.102 -15.115  1.00  0.00           N
ATOM      7  CA  GLY A   2      -3.262  -2.499 -16.313  1.00  0.00           C
ATOM      8  C   GLY A   2      -3.961  -3.660 -17.011  1.00  0.00           C
ATOM      9  O   GLY A   2      -4.016  -3.716 -18.240  1.00  0.00           O
"""

BETA_BACKBONE = """
ATOM      1  N   GLY A   1      27.961   0.504   1.988  1.00  0.00           N
ATOM      2  CA  GLY A   1      29.153   0.205   2.773  1.00  0.00           C
ATOM      3  C   GLY A   1      30.420   0.562   2.003  1.00  0.00           C
ATOM      4  O   GLY A   1      30.753  -0.077   1.005  1.00  0.00           O
ATOM      6  N   GLY A   2      31.123   1.587   2.474  1.00  0.00           N
ATOM      7  CA  GLY A   2      32.355   2.031   1.832  1.00  0.00           C
ATOM      8  C   GLY A   2      33.552   1.851   2.758  1.00  0.00           C
ATOM      9  O   GLY A   2      33.675   2.539   3.772  1.00  0.00           O
"""

HELIX_ROTAMERS = {
    "ALA": "ATOM      1  N   ALA A   1     140.893 179.647 182.210  1.00  0.00           N  \nATOM      2  CA  ALA A   1     140.230 178.969 181.101  1.00  0.00           C  \nATOM      3  C   ALA A   1     140.375 179.758 179.806  1.00  0.00           C  \nATOM      4  O   ALA A   1     140.615 179.179 178.740  1.00  0.00           O  \nATOM      5  CB  ALA A   1     138.756 178.743 181.430  1.00  0.00           C  \n",
    "ARG": "ATOM      1  N   ARG A   1     145.207 177.880 170.323  1.00  0.00           N  \nATOM      2  CA  ARG A   1     145.178 178.673 169.099  1.00  0.00           C  \nATOM      3  C   ARG A   1     146.586 178.920 168.576  1.00  0.00           C  \nATOM      4  O   ARG A   1     146.836 178.822 167.369  1.00  0.00           O  \nATOM      5  CB  ARG A   1     144.458 179.997 169.348  1.00  0.00           C  \nATOM      6  CG  ARG A   1     144.502 180.953 168.172  1.00  0.00           C  \nATOM      7  CD  ARG A   1     143.701 182.210 168.443  1.00  0.00           C  \nATOM      8  NE  ARG A   1     143.874 183.198 167.386  1.00  0.00           N  \nATOM      9  CZ  ARG A   1     143.179 183.217 166.257  1.00  0.00           C  \nATOM     10  NH1 ARG A   1     142.249 182.312 166.004  1.00  0.00           N  \nATOM     11  NH2 ARG A   1     143.424 184.167 165.360  1.00  0.00           N  \n",
    "ASN": "ATOM      1  N   ASN A   1     150.336 174.937 161.065  1.00  0.00           N  \nATOM      2  CA  ASN A   1     150.174 175.420 159.698  1.00  0.00           C  \nATOM      3  C   ASN A   1     151.468 176.024 159.165  1.00  0.00           C  \nATOM      4  O   ASN A   1     151.800 175.851 157.986  1.00  0.00           O  \nATOM      5  CB  ASN A   1     149.039 176.440 159.634  1.00  0.00           C  \nATOM      6  CG  ASN A   1     147.686 175.822 159.923  1.00  0.00           C  \nATOM      7  OD1 ASN A   1     147.083 175.188 159.058  1.00  0.00           O  \nATOM      8  ND2 ASN A   1     147.201 176.005 161.145  1.00  0.00           N  \n",
    "ASP": "ATOM      1  N   ASP A   1     142.837 181.888 179.273  1.00  0.00           N  \nATOM      2  CA  ASP A   1     144.264 181.868 178.969  1.00  0.00           C  \nATOM      3  C   ASP A   1     144.670 180.542 178.335  1.00  0.00           C  \nATOM      4  O   ASP A   1     145.477 180.516 177.399  1.00  0.00           O  \nATOM      5  CB  ASP A   1     145.064 182.134 180.246  1.00  0.00           C  \nATOM      6  CG  ASP A   1     146.530 182.449 179.982  1.00  0.00           C  \nATOM      7  OD1 ASP A   1     147.062 182.081 178.914  1.00  0.00           O  \nATOM      8  OD2 ASP A   1     147.159 183.079 180.859  1.00  0.00           O  \n",
    "CYS": "ATOM      1  N   CYS A   1     178.475 187.332 138.876  1.00  0.00           N  \nATOM      2  CA  CYS A   1     179.226 186.152 139.286  1.00  0.00           C  \nATOM      3  C   CYS A   1     180.388 186.480 140.214  1.00  0.00           C  \nATOM      4  O   CYS A   1     180.997 185.556 140.763  1.00  0.00           O  \nATOM      5  CB  CYS A   1     179.732 185.394 138.056  1.00  0.00           C  \nATOM      6  SG  CYS A   1     180.862 186.315 136.996  1.00  0.00           S  \n",
    "GLN": "ATOM      1  N   GLN A   1     158.568 180.258 126.473  1.00  0.00           N  \nATOM      2  CA  GLN A   1     158.871 178.876 126.119  1.00  0.00           C  \nATOM      3  C   GLN A   1     158.517 177.925 127.255  1.00  0.00           C  \nATOM      4  O   GLN A   1     157.958 176.847 127.020  1.00  0.00           O  \nATOM      5  CB  GLN A   1     160.346 178.742 125.747  1.00  0.00           C  \nATOM      6  CG  GLN A   1     160.686 177.434 125.064  1.00  0.00           C  \nATOM      7  CD  GLN A   1     159.960 177.273 123.746  1.00  0.00           C  \nATOM      8  OE1 GLN A   1     158.990 176.522 123.646  1.00  0.00           O  \nATOM      9  NE2 GLN A   1     160.425 177.983 122.724  1.00  0.00           N  \n",
    "GLU": "ATOM      1  N   GLU A   1     140.221 181.081 179.878  1.00  0.00           N  \nATOM      2  CA  GLU A   1     140.462 181.920 178.709  1.00  0.00           C  \nATOM      3  C   GLU A   1     141.928 181.879 178.296  1.00  0.00           C  \nATOM      4  O   GLU A   1     142.242 181.850 177.101  1.00  0.00           O  \nATOM      5  CB  GLU A   1     140.028 183.355 178.999  1.00  0.00           C  \nATOM      6  CG  GLU A   1     138.551 183.515 179.303  1.00  0.00           C  \nATOM      7  CD  GLU A   1     137.722 183.737 178.059  1.00  0.00           C  \nATOM      8  OE1 GLU A   1     138.224 184.382 177.116  1.00  0.00           O  \nATOM      9  OE2 GLU A   1     136.565 183.270 178.023  1.00  0.00           O  \n",
    "GLY": "ATOM      1  N   GLY A   1     156.198 175.742 155.360  1.00  0.00           N  \nATOM      2  CA  GLY A   1     157.603 175.957 155.067  1.00  0.00           C  \nATOM      3  C   GLY A   1     158.311 174.759 154.477  1.00  0.00           C  \nATOM      4  O   GLY A   1     159.369 174.923 153.860  1.00  0.00           O  \n",
    "HIS": "ATOM      1  N   HIS A   1     179.387 194.583 124.939  1.00  0.00           N  \nATOM      2  CA  HIS A   1     180.338 194.799 126.024  1.00  0.00           C  \nATOM      3  C   HIS A   1     179.621 195.065 127.340  1.00  0.00           C  \nATOM      4  O   HIS A   1     180.048 194.582 128.395  1.00  0.00           O  \nATOM      5  CB  HIS A   1     181.274 195.959 125.685  1.00  0.00           C  \nATOM      6  CG  HIS A   1     182.405 195.581 124.781  1.00  0.00           C  \nATOM      7  ND1 HIS A   1     182.433 195.914 123.444  1.00  0.00           N  \nATOM      8  CE1 HIS A   1     183.546 195.454 122.901  1.00  0.00           C  \nATOM      9  NE2 HIS A   1     184.242 194.836 123.839  1.00  0.00           N  \nATOM     10  CD2 HIS A   1     183.551 194.902 125.024  1.00  0.00           C  \n",
    "ILE": "ATOM      1  N   ILE A   1     148.368 177.422 165.109  1.00  0.00           N  \nATOM      2  CA  ILE A   1     149.034 178.283 164.136  1.00  0.00           C  \nATOM      3  C   ILE A   1     150.403 177.719 163.775  1.00  0.00           C  \nATOM      4  O   ILE A   1     150.815 177.750 162.609  1.00  0.00           O  \nATOM      5  CB  ILE A   1     149.133 179.720 164.680  1.00  0.00           C  \nATOM      6  CG1 ILE A   1     147.765 180.403 164.644  1.00  0.00           C  \nATOM      7  CD1 ILE A   1     147.747 181.760 165.304  1.00  0.00           C  \nATOM      8  CG2 ILE A   1     150.139 180.533 163.887  1.00  0.00           C  \n",
    "LEU": "ATOM      1  N   LEU A   1     152.205 176.743 160.014  1.00  0.00           N  \nATOM      2  CA  LEU A   1     153.448 177.367 159.572  1.00  0.00           C  \nATOM      3  C   LEU A   1     154.487 176.324 159.178  1.00  0.00           C  \nATOM      4  O   LEU A   1     155.145 176.456 158.140  1.00  0.00           O  \nATOM      5  CB  LEU A   1     153.999 178.274 160.671  1.00  0.00           C  \nATOM      6  CG  LEU A   1     153.162 179.493 161.055  1.00  0.00           C  \nATOM      7  CD1 LEU A   1     153.977 180.434 161.921  1.00  0.00           C  \nATOM      8  CD2 LEU A   1     152.653 180.206 159.816  1.00  0.00           C  \n",
    "LYS": "ATOM      1  N   LYS A   1     144.114 179.432 178.827  1.00  0.00           N  \nATOM      2  CA  LYS A   1     144.437 178.127 178.258  1.00  0.00           C  \nATOM      3  C   LYS A   1     144.016 178.037 176.798  1.00  0.00           C  \nATOM      4  O   LYS A   1     144.783 177.561 175.954  1.00  0.00           O  \nATOM      5  CB  LYS A   1     143.768 177.019 179.070  1.00  0.00           C  \nATOM      6  CG  LYS A   1     144.591 176.519 180.238  1.00  0.00           C  \nATOM      7  CD  LYS A   1     143.890 175.378 180.952  1.00  0.00           C  \nATOM      8  CE  LYS A   1     142.734 175.887 181.794  1.00  0.00           C  \nATOM      9  NZ  LYS A   1     142.139 174.812 182.632  1.00  0.00           N  \n",
    "MET": "ATOM      1  N   MET A   1     147.935 175.628 167.177  1.00  0.00           N  \nATOM      2  CA  MET A   1     147.360 175.327 165.870  1.00  0.00           C  \nATOM      3  C   MET A   1     147.973 176.192 164.777  1.00  0.00           C  \nATOM      4  O   MET A   1     148.083 175.749 163.628  1.00  0.00           O  \nATOM      5  CB  MET A   1     145.844 175.513 165.912  1.00  0.00           C  \nATOM      6  CG  MET A   1     145.108 174.927 164.721  1.00  0.00           C  \nATOM      7  SD  MET A   1     143.344 174.720 165.036  1.00  0.00           S  \nATOM      8  CE  MET A   1     142.648 175.365 163.517  1.00  0.00           C  \n",
    "PHE": "ATOM      1  N   PHE A   1     177.584 190.808 133.557  1.00  0.00           N  \nATOM      2  CA  PHE A   1     177.922 189.439 133.933  1.00  0.00           C  \nATOM      3  C   PHE A   1     179.003 189.416 135.005  1.00  0.00           C  \nATOM      4  O   PHE A   1     178.926 188.635 135.960  1.00  0.00           O  \nATOM      5  CB  PHE A   1     178.370 188.652 132.700  1.00  0.00           C  \nATOM      6  CG  PHE A   1     178.516 187.175 132.939  1.00  0.00           C  \nATOM      7  CD1 PHE A   1     179.720 186.643 133.368  1.00  0.00           C  \nATOM      8  CE1 PHE A   1     179.857 185.287 133.585  1.00  0.00           C  \nATOM      9  CZ  PHE A   1     178.788 184.445 133.371  1.00  0.00           C  \nATOM     10  CE2 PHE A   1     177.584 184.960 132.942  1.00  0.00           C  \nATOM     11  CD2 PHE A   1     177.452 186.317 132.726  1.00  0.00           C  \n",
    "PRO": "ATOM      1  N   PRO A   1     191.318 150.020 180.056  1.00  0.00           N  \nATOM      2  CA  PRO A   1     192.071 149.923 181.316  1.00  0.00           C  \nATOM      3  C   PRO A   1     191.625 148.775 182.209  1.00  0.00           C  \nATOM      4  O   PRO A   1     192.463 148.066 182.776  1.00  0.00           O  \nATOM      5  CB  PRO A   1     191.808 151.277 181.989  1.00  0.00           C  \nATOM      6  CG  PRO A   1     191.363 152.175 180.891  1.00  0.00           C  \nATOM      7  CD  PRO A   1     190.577 151.290 179.983  1.00  0.00           C  \n",
    "SER": "ATOM      1  N   SER A   1     147.520 179.244 169.472  1.00  0.00           N  \nATOM      2  CA  SER A   1     148.909 179.431 169.068  1.00  0.00           C  \nATOM      3  C   SER A   1     149.520 178.130 168.564  1.00  0.00           C  \nATOM      4  O   SER A   1     150.272 178.131 167.583  1.00  0.00           O  \nATOM      5  CB  SER A   1     149.722 179.987 170.234  1.00  0.00           C  \nATOM      6  OG  SER A   1     151.100 179.718 170.062  1.00  0.00           O  \n",
    "THR": "ATOM      1  N   THR A   1     161.755 185.528 125.043  1.00  0.00           N  \nATOM      2  CA  THR A   1     161.509 184.311 124.277  1.00  0.00           C  \nATOM      3  C   THR A   1     161.507 183.080 125.176  1.00  0.00           C  \nATOM      4  O   THR A   1     160.646 182.203 125.036  1.00  0.00           O  \nATOM      5  CB  THR A   1     162.555 184.168 123.170  1.00  0.00           C  \nATOM      6  OG1 THR A   1     162.334 185.169 122.169  1.00  0.00           O  \nATOM      7  CG2 THR A   1     162.475 182.792 122.525  1.00  0.00           C  \n",
    "TRP": "ATOM      1  N   TRP A   1     170.626 155.391 172.714  1.00  0.00           N  \nATOM      2  CA  TRP A   1     171.095 155.042 174.049  1.00  0.00           C  \nATOM      3  C   TRP A   1     170.042 154.281 174.844  1.00  0.00           C  \nATOM      4  O   TRP A   1     170.392 153.419 175.657  1.00  0.00           O  \nATOM      5  CB  TRP A   1     171.512 156.302 174.809  1.00  0.00           C  \nATOM      6  CG  TRP A   1     172.668 157.025 174.190  1.00  0.00           C  \nATOM      7  CD1 TRP A   1     172.642 157.810 173.076  1.00  0.00           C  \nATOM      8  NE1 TRP A   1     173.896 158.304 172.815  1.00  0.00           N  \nATOM      9  CE2 TRP A   1     174.763 157.840 173.769  1.00  0.00           C  \nATOM     10  CZ2 TRP A   1     176.127 158.064 173.928  1.00  0.00           C  \nATOM     11  CH2 TRP A   1     176.745 157.463 174.989  1.00  0.00           C  \nATOM     12  CZ3 TRP A   1     176.038 156.654 175.884  1.00  0.00           C  \nATOM     13  CE3 TRP A   1     174.684 156.430 175.729  1.00  0.00           C  \nATOM     14  CD2 TRP A   1     174.024 157.031 174.653  1.00  0.00           C  \n",
    "TYR": "ATOM      1  N   TYR A   1     181.590 180.389 111.254  1.00  0.00           N  \nATOM      2  CA  TYR A   1     180.607 180.452 112.330  1.00  0.00           C  \nATOM      3  C   TYR A   1     180.970 179.543 113.499  1.00  0.00           C  \nATOM      4  O   TYR A   1     180.581 179.824 114.638  1.00  0.00           O  \nATOM      5  CB  TYR A   1     179.221 180.089 111.789  1.00  0.00           C  \nATOM      6  CG  TYR A   1     178.186 179.822 112.860  1.00  0.00           C  \nATOM      7  CD1 TYR A   1     177.605 180.866 113.566  1.00  0.00           C  \nATOM      8  CE1 TYR A   1     176.662 180.628 114.545  1.00  0.00           C  \nATOM      9  CZ  TYR A   1     176.287 179.334 114.830  1.00  0.00           C  \nATOM     10  OH  TYR A   1     175.347 179.095 115.805  1.00  0.00           O  \nATOM     11  CE2 TYR A   1     176.847 178.279 114.143  1.00  0.00           C  \nATOM     12  CD2 TYR A   1     177.791 178.526 113.165  1.00  0.00           C  \n",
    "VAL": "ATOM      1  N   VAL A   1     146.354 179.257 174.162  1.00  0.00           N  \nATOM      2  CA  VAL A   1     147.549 178.470 173.878  1.00  0.00           C  \nATOM      3  C   VAL A   1     147.298 177.503 172.727  1.00  0.00           C  \nATOM      4  O   VAL A   1     148.144 177.347 171.838  1.00  0.00           O  \nATOM      5  CB  VAL A   1     148.018 177.736 175.145  1.00  0.00           C  \nATOM      6  CG1 VAL A   1     149.250 176.902 174.850  1.00  0.00           C  \nATOM      7  CG2 VAL A   1     148.300 178.730 176.255  1.00  0.00           C  \n",
}


if __name__ == "__main__":
    breakpoint()
    print("success")
