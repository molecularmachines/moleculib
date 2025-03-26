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
    if residue == 'HSD' or residue == 'HSE' or residue == 'HSP':
        residue = 'HIS'
    return _index(all_residues, residue)


atom_to_residues_index, atom_to_residues_mask = zip(
    *list(map(_atom_to_all_residues_index, all_atoms))
)
atom_to_residues_index = np.array(atom_to_residues_index)
atom_to_residues_mask = np.array(atom_to_residues_mask)


if __name__ == "__main__":
    breakpoint()
    print("success")
