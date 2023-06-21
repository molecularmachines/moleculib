from collections import OrderedDict
from typing import List

import numpy as np
from ordered_set import OrderedSet

UNK_TOKEN = 1
MAX_DNA_ATOMS = 11 #5 carbon atoms 8 hydrogen atoms 1 nitrogen atom (in the base) 
                   # 1 phosphorus atom (in the phosphate group) 4 oxygen atoms (in the sugar and phosphate group)
MAX_RES_ATOMS = 14
# from https://x3dna.org/articles/name-of-base-atoms-in-pdb-formats#:~:text=Canonical%20bases%20(A%2C%20C%2C,%2C%20C8%2C%20N9)%20respectively.

#ensure order and keep track of what atoms are present or missing
base_atoms_per_nuc = OrderedDict(
        A=["N1","C2","N3", "C4", "C5", "C6", "N7", "C8", "N9"], #calman had N6
        U=["N1", "C2", "O2", "N3", "C4", "C5", "C6"], #
        T=["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C5M", "C6"], #Calman Didn't have C5M but had:
        #T=['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6']
        G=["N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"], #same
        C=["N1", "C2", "O2", "N3", "C4", "C5", "C6"] #Calman had extra N4
)


backbone_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'","P", "O1P", "O2P","O3P","O2'", "O3'", "O4'", "O5'"] ########TODO TODO check it

### TODO: Should it be the same as below (from alphabet protein)?
backbone_chemistry = dict(
    bonds=[["N", "CA"], ["CA", "C"], ["C", "O"]],
    angles=[["CA", "C", "O"], ["N", "CA", "C"]],
    dihedrals=[["N", "CA", "C", "O"]],
    flippable=[],
)
special_tokens = ["PAD", "UNK"] #what do whese represent?

atoms_per_nuc = OrderedDict()
atoms_per_nuc["PAD"] = []
atoms_per_nuc["UNK"] = backbone_atoms
##TODO check the DUPLICATES situation
#for every nuc we add the base and backbone atoms
for nuc, base_atoms in base_atoms_per_nuc.items():
    atoms_per_nuc[nuc] = backbone_atoms + base_atoms

all_atoms = list(OrderedSet(sum(list(atoms_per_nuc.values()), [])))
all_atoms = special_tokens + all_atoms
all_atoms_tokens = np.arange(len(all_atoms))

elements = list(OrderedSet([atom[0] for atom in all_atoms]))
all_atoms_elements = np.array([elements.index(atom[0]) for atom in all_atoms])
# all_atoms_radii = np.array(
#     [
#         (van_der_walls_radii[atom[0]] if (atom[0] in van_der_walls_radii) else 0.0)
#         for atom in all_atoms
#     ]
# )

all_nucs = list(base_atoms_per_nuc.keys())
all_nucs = special_tokens + all_nucs
all_nucs_tokens = np.arange(len(all_nucs))
all_nucs_atom_mask = np.array(
    [
        ([1] * len(atoms) + [0] * (14 - len(atoms)))
        for (_, atoms) in atoms_per_nuc.items()
    ]
).astype(np.bool_)

def _atom_to_all_nucs_index(atom):
    def _atom_to_nuc_index(nuc):
        nuc_atoms = atoms_per_nuc[nuc]
        mask = atom in nuc_atoms
        index = nuc_atoms.index(atom) if mask else 0
        return index, mask

    indices, masks = zip(*list(map(_atom_to_nuc_index, all_nucs)))
    return np.array(indices), np.array(masks)


def _index(lst: List[str], item: str) -> int:
    try:
        index = lst.index(item)
    except ValueError:
        index = UNK_TOKEN  # UNK
    return index


def atom_index(atom: str) -> int:
    return _index(all_atoms, atom)

def get_nucleotide_index(nucleotide: str) -> int:
    return _index(all_nucs, nucleotide)

atom_to_nucs_index, atom_to_nucs_mask = zip(
    *list(map(_atom_to_all_nucs_index, all_atoms))
)
atom_to_nucs_index = np.array(atom_to_nucs_index)
atom_to_nucs_mask = np.array(atom_to_nucs_mask)