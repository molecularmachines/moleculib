from collections import OrderedDict
from typing import List

import numpy as np
from ordered_set import OrderedSet

# UNK_TOKEN = 1
#5 carbon atoms 8 hydrogen atoms 1 nitrogen atom (in the base) 
# 1 phosphorus atom (in the phosphate group) 4 oxygen atoms (in the sugar and phosphate group)
MAX_RES_ATOMS = 14
# from https://x3dna.org/articles/name-of-base-atoms-in-pdb-formats#:~:text=Canonical%20bases%20(A%2C%20C%2C,%2C%20C8%2C%20N9)%20respectively.

#ensure order and keep track of what atoms are present or missing
base_atoms_per_nuc = OrderedDict(
        A=["N1","C2","N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"], # GREEN
        U=["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"], #RED
        RT=["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C5M", "C6", "C7"], #Calman Didn't have C5M  'C5' instead
        G=["N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"], #light pink (has N9, O6)
        C=["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"], # pink (has o2)
        I = [], #from gpt: I=["N1", "C2", "N3", "C4", "C5", "C6"]
        #NOTE: in the file where I took the atoms, DA's atoms are N1A, C2A, etc.
        #but from working with the data I saw that they're regular without 'A'
        DA = ["N1","C2", "N3", "C4", "C5", "N6", "C6", "N7","C8","N9"],#olive GREEN
        DC =["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"], #tourcize
        DG =["N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"], #light blue-violet
        DI =[],
        #NOTE: actually didn't see C5M in the data itself but leave it in case it will come up
        DT = ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C5M", "C6", "C7"], #purple
        DU =["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"] #mutagenic U (doesn't suppose to be a DNA nuc but RNA nuc)
)


backbone_atoms_DNA = ["P", "OP1", "OP2", "OP3", "C1'", "C2'", "C3'", "C4'", "C5'",  "O3'", "O4'", "O5'"] #NOTE: add "" for carbons and Oxygens as in the file
backbone_atoms_RNA = ["P", "OP1", "OP2",  "OP3","C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'"]  


special_tokens = [  "PAD", "UNK"] #what do whese represent?

atoms_per_nuc = OrderedDict()
##TODO check the DUPLICATES situation
#for every nuc we add the base and backbone atoms
for nuc, base_atoms in base_atoms_per_nuc.items():
    if nuc in ['A', 'U', 'RT', 'G', 'C', 'I']: #RNA
        atoms_per_nuc[nuc] = backbone_atoms_RNA+ base_atoms 
    else:
        atoms_per_nuc[nuc] = backbone_atoms_DNA+ base_atoms
atoms_per_nuc["PAD"] = []
atoms_per_nuc["UNK"] = [] # backbone_atoms

MAX_DNA_ATOMS = max([len(atoms) for atoms in atoms_per_nuc.values()])###==24 ###NOTE MAYBE MORE? 31

all_atoms = list(OrderedSet(sum(list(atoms_per_nuc.values()), [])))
all_atoms = special_tokens + all_atoms   ###NOTE SWITCHEDDD
# print("all atoms", all_atoms)
all_atoms_tokens = np.arange(len(all_atoms))

elements = list(OrderedSet([atom[0] for atom in all_atoms]))
all_atoms_elements = np.array([elements.index(atom[0]) for atom in all_atoms])


all_nucs = list(base_atoms_per_nuc.keys())
all_nucs = all_nucs + special_tokens

all_nucs_tokens = np.arange(len(all_nucs))
all_nucs_atom_mask = np.array(
    [
        ([1] * len(atoms) + [0] * (MAX_DNA_ATOMS - len(atoms)))
        for (_, atoms) in atoms_per_nuc.items()
    ]
).astype(np.bool_)

all_nucs_atom_tokens = np.array(
    [
        ([all_atoms.index(atom) for atom in atoms] + [0] * (24 - len(atoms)))
        for (_, atoms) in atoms_per_nuc.items()
    ]
)

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
        index = 1 #UNK_TOKEN  # UNK ###NOTE DO 30
    return index


def atom_index(atom: str) -> int:
    return _index(all_atoms, atom)

def get_nucleotide_index(nucleotide: str) -> int: #get index straight from here to make sure UNK index is set to 13 and not 30.abs
    try:
        index = all_nucs.index(nucleotide)
    except ValueError:
        index = 13 #UNK_TOKEN  # UNK
    return index
    # return _index(all_nucs, nucleotide)

atom_to_nucs_index, atom_to_nucs_mask = zip(
    *list(map(_atom_to_all_nucs_index, all_atoms))
)
atom_to_nucs_index = np.array(atom_to_nucs_index)
atom_to_nucs_mask = np.array(atom_to_nucs_mask)


# # print(get_nucleotide_index("PAD"))
# for i in backbone_atoms_RNA:
#     print(i, atom_index(i))
