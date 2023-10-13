import numpy as np
from Bio.PDB import parse_pdb_header
from biotite.database import rcsb
from biotite.sequence import ProteinSequence, NucleotideSequence, GeneralSequence, Alphabet
from biotite.structure import (
    apply_chain_wise,
    apply_residue_wise,
    get_chain_count,
    get_residue_count,
    get_residues,
    get_chains,
    spread_chain_wise,
    spread_residue_wise,
)
from biotite.structure import filter_nucleotides
import os
import biotite.structure.io.mmtf as mmtf

import sys
sys.path.append('.')

# from alphabet import (
#     all_atoms,
#     all_nucs,
#     backbone_atoms_DNA,
#     backbone_atoms_RNA,
#     atom_index,
#     atom_to_nucs_index,
#     get_nucleotide_index,
#     MAX_DNA_ATOMS
# )

## alphabet:
from collections import OrderedDict
from typing import List

import numpy as np
from ordered_set import OrderedSet

UNK_TOKEN = 1
#5 carbon atoms 8 hydrogen atoms 1 nitrogen atom (in the base) 
# 1 phosphorus atom (in the phosphate group) 4 oxygen atoms (in the sugar and phosphate group)
MAX_RES_ATOMS = 14
# from https://x3dna.org/articles/name-of-base-atoms-in-pdb-formats#:~:text=Canonical%20bases%20(A%2C%20C%2C,%2C%20C8%2C%20N9)%20respectively.

#ensure order and keep track of what atoms are present or missing
base_atoms_per_nuc = OrderedDict(
        A=["N1","C2","N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"], # GREEN
        U=["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"], #RED
        RT=["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C5M", "C6"], #Calman Didn't have C5M  'C5' instead
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
        DT = ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C5M", "C6"], #purple
        DU =["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"] #mutagenic U (doesn't suppose to be a DNA nuc but RNA nuc)
)

backbone_atoms_DNA = ["C1'", "C2'", "C3'", "C4'", "C5'","P", "O1P", "O2P","O3P", "O3'", "O4'", "O5'"] #NOTE: add "" for carbons and Oxygens as in the file
backbone_atoms_RNA = ["C1'", "C2'", "C3'", "C4'", "C5'","P", "O1P", "O2P","O3P","O2'", "O3'", "O4'", "O5'"]  

### TODO: Should it be the same as below (from alphabet protein)?
# backbone_chemistry = dict(
#     bonds=[["N", "CA"], ["CA", "C"], ["C", "O"]],
#     angles=[["CA", "C", "O"], ["N", "CA", "C"]],
#     dihedrals=[["N", "CA", "C", "O"]],
#     flippable=[],
# )
special_tokens = ["PAD", "UNK"] #what do whese represent?

atoms_per_nuc = OrderedDict()
atoms_per_nuc["PAD"] = []
atoms_per_nuc["UNK"] = [] # backbone_atoms
##TODO check the DUPLICATES situation
#for every nuc we add the base and backbone atoms
for nuc, base_atoms in base_atoms_per_nuc.items():
    if nuc in ['A', 'U', 'RT', 'G', 'C', 'I']: #RNA
        atoms_per_nuc[nuc] = base_atoms + backbone_atoms_RNA
    else:
        atoms_per_nuc[nuc] = base_atoms + backbone_atoms_DNA
MAX_DNA_ATOMS = max([len(atoms) for atoms in atoms_per_nuc.values()])###==24

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
        ([1] * len(atoms) + [0] * (MAX_DNA_ATOMS - len(atoms)))
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

#### end of alphabet

# from utils import  pdb_to_atom_array
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

##END OF UTILS

class NucleicDatum:
    """
    Incorporates Biotite nucleic sequence data 
    # and reshapes atom arrays to residue-based representation
    """

    def __init__(
        self,
        idcode: str,
        resolution: float,
        sequence: NucleotideSequence,
        nuc_token: np.ndarray,
        nuc_index: np.ndarray,
        nuc_mask: np.ndarray,
        chain_token: np.ndarray,
        atom_token: np.ndarray,
        atom_coord: np.ndarray,
        atom_mask: np.ndarray,
    ):
        self.idcode = idcode
        self.resolution = resolution
        self.sequence = str(sequence)
        self.nuc_token = nuc_token
        self.nuc_index = nuc_index #do we need that
        self.nuc_mask = nuc_mask
        self.chain_token = chain_token
        self.atom_token = atom_token
        self.atom_coord = atom_coord
        self.atom_mask = atom_mask
    

    def __len__(self):
        return len(self.nuc_index)

    @classmethod
    def _extract_reshaped_atom_attr(
        cls,
        atom_array,
        atom_alphabet=all_atoms,
        atom_to_indices=atom_to_nucs_index,
        attrs=["coord", "token"],
    ):
        residue_count = get_residue_count(atom_array)
        extraction = dict()
        mask = np.zeros((residue_count, MAX_DNA_ATOMS)).astype(bool)
        #creates array of zeroes in the desired shape to 
        #fill with tokens and coords
        for attr in attrs:
            attr_shape = getattr(atom_array, attr).shape
            if len(attr_shape) == 1:
                attr_reshape = np.zeros((residue_count, MAX_DNA_ATOMS))
            else:
                attr_reshape = np.zeros((residue_count, MAX_DNA_ATOMS, attr_shape[-1]))
            extraction[attr] = attr_reshape

        #NOTE: explored the different chains specifically for 5F9R complex:
        #the following 3 funcs are for this exploration: 
        def get_repetition_lengths(lst):
            lengths = []
            count = 1
            for i in range(1, len(lst)):
                if lst[i] == lst[i-1]:
                    count += 1
                else:
                    lengths.append(count)
                    count = 1
            lengths.append(count)
            return lengths

        def check_strictly_increasing(lst):
            for i in range(1, len(lst)):
                if lst[i] < lst[i-1]:
                    return False
            return True
        
        def check_res(lst):
            res =set()
            for i in lst:
                if i not in res:
                    res.add(i)
            return res
        
        #NOTE: explored the different chains specifically for 5F9R complex:
        # chainA = atom_array[:2462]
        # chainC = atom_array[2462:2462+601]
        # chainD = atom_array[2462+601:]
        
        # print(check_strictly_increasing(atom_array.res_id[2462+601:]))
        # repetition_lengths = get_repetition_lengths(atom_array.chain_id) #[2462, 601, 396]
        # print(repetition_lengths)
        # print(len(atom_array.res_name))

        def _atom_slice(atom_name, atom_array, atom_token):
            atom_array_ = atom_array[(atom_array.atom_name == atom_name)]
            # kill pads and kill unks that are not backbone
            atom_array_ = atom_array_[(atom_array_.residue_token > 0)]

            if atom_name not in backbone_atoms_RNA:
                atom_array_ = atom_array_[(atom_array_.residue_token > 1)]

            res_tokens, seq_id = atom_array_.residue_token, atom_array_.seq_uid
            atom_indices = atom_to_indices[atom_token][res_tokens]

            for attr in attrs:
                attr_tensor = getattr(atom_array_, attr)
                extraction[attr][seq_id, atom_indices, ...] = attr_tensor
            mask[seq_id, atom_indices] = True

        for atom_name in atom_alphabet:

            atom_token = atom_alphabet.index(atom_name)
            _atom_slice(atom_name, atom_array, atom_token) ####NOTE
        
        return extraction, mask


    @classmethod
    def empty_nuc(cls):
        return cls(
            idcode="",
            resolution=0.0,
            sequence=NucleotideSequence(""),
            nuc_index=np.array([]),
            nuc_token=np.array([]),
            nuc_mask=np.array([]),
            chain_token=np.array([]),
            atom_token=np.array([]),
            atom_coord=np.array([]),
            atom_mask=np.array([])
            )

    @classmethod
    def from_filepath(cls, filepath):
        atom_array =  pdb_to_atom_array(filepath) #filters pdb to only nucleotides
        header = parse_pdb_header(filepath)    
        return cls.from_atom_array(atom_array, header=header)

    @classmethod
    def fetch_pdb_id(cls, id, save_path=None):
        filepath = rcsb.fetch(id, "pdb", save_path)
        return cls.from_filepath(filepath)

    @classmethod
    def from_atom_array(
        cls,
        atom_array,
        header,
        ):
        """
        Reshapes atom array to residue-indexed representation to
        build a protein datum.
        """

        if atom_array.array_length() == 0:
            return cls.empty_nuc()
        
        _, res_names = get_residues(atom_array)
        res_names = [
            ("UNK" if (name not in all_nucs) else name) for name in res_names
        ]

        sequence = GeneralSequence(Alphabet(all_nucs), list(res_names))
        # breakpoint()
        # index residues globally
        atom_array.add_annotation("seq_uid", int)
        atom_array.seq_uid = spread_residue_wise(
            atom_array, np.arange(0, get_residue_count(atom_array))
        )

        # tokenize atoms
        atom_array.add_annotation("token", int)
        atom_array.token = np.array(
            list(map(lambda atom: atom_index(atom), atom_array.atom_name))
        )

        # tokenize residues
        residue_token = np.array(
            list(map(lambda res: get_nucleotide_index(res), atom_array.res_name))
        )
        
        residue_mask = np.ones_like(residue_token).astype(bool)

        atom_array.add_annotation("residue_token", int)
        atom_array.residue_token = residue_token
        chain_token = spread_chain_wise(
            atom_array, np.arange(0, get_chain_count(atom_array))
        )

        # count number of residues per chain
        # and index residues per chain using cumulative sum
        atom_array.add_annotation("res_uid", int)

        def _count_residues_per_chain(chain_atom_array, axis=0):
            return get_residue_count(chain_atom_array)

        chain_res_sizes = apply_chain_wise(
            atom_array, atom_array, _count_residues_per_chain, axis=0
        )
        chain_res_cumsum = np.cumsum([0] + list(chain_res_sizes[:-1]))
        atom_array.res_uid = atom_array.res_id + chain_res_cumsum[chain_token]
        
        # reshape atom attributes to residue-based representation
        # with the correct ordering
        # [N * 14, ...] -> [N, 14, ...]
        atom_extract, atom_mask = cls._extract_reshaped_atom_attr(
            atom_array, atom_alphabet=all_atoms, atom_to_indices=atom_to_nucs_index, attrs=["coord", "token"]
        )



        atom_extract = dict(
            map(lambda kv: (f"atom_{kv[0]}", kv[1]), atom_extract.items())
        )

        # pool residue attributes and create residue features
        # [N * 14, ...] -> [N, ...]
        def _pool_residue_token(atom_residue_tokens, axis=0):
            representative = atom_residue_tokens[0]
            return representative

        def _reshape_residue_attr(attr):
            return apply_residue_wise(atom_array, attr, _pool_residue_token, axis=0)

        residue_token = _reshape_residue_attr(residue_token)
        residue_index = np.arange(0, residue_token.shape[0])

        residue_mask = _reshape_residue_attr(residue_mask)
        residue_mask = residue_mask & (atom_extract["atom_coord"].sum((-1, -2)) != 0)

        chain_token = _reshape_residue_attr(chain_token)
        return cls(
            idcode=header["idcode"],
            sequence=sequence,
            resolution=header["resolution"],
            nuc_token=residue_token,
            nuc_index=residue_index,
            nuc_mask=residue_mask,
            chain_token=chain_token,
            **atom_extract,
            atom_mask=atom_mask,
        )






import plotly.graph_objects as go
import plotly.offline as pyo
def _scatter_coord(name, coord, color='black', visible=True):
    sc_coords = []
    x, y, z = coord.T
    data = [
        go.Scatter3d(
            name=name + " coord",
            x=x,
            y=y,
            z=z,
            marker=dict(
                size=7,
                colorscale="Viridis",
            ),
            
            hovertemplate="<b>%{text}</b><extra></extra>",
            text=np.arange(0, len(coord)),
            
            line=dict(color=color, width=4),
            visible="legendonly" if not visible else True,
        )
    ]
    fig = go.Figure(data=data)
    fig.update_layout(
        width=650,
        height=750,
    )
    fig.update_scenes(
        xaxis_visible=False, 
        yaxis_visible=False, 
        zaxis_visible=False
    )
    
    return fig

# np.array(list(map(lambda atom: atom_index(atom), atom_array.atom_name)))
rna_res_names = ['A', 'U', 'RT', 'G', 'C', 'I']
dna_res_names = ['DA', 'DU', 'DT', 'DG', 'DC', 'DI']
dna_res_tokens = list(map(lambda res: get_nucleotide_index(res), dna_res_names))
rna_res_tokens = list(map(lambda res: get_nucleotide_index(res), rna_res_names))


if __name__ == '__main__':
    dna_datum = NucleicDatum.fetch_pdb_id('5F9R')    
    # breakpoint()
    ##DNADATUM: str,
        # resolution: float,
        # sequence: NucleotideSequence,
        # nuc_token: np.ndarray, #tokenize nucs (0 to ~14 or so, ie options of nucs)
        # nuc_index: np.ndarray, #index each nuc from 0 to len of nucleotides in the datum
        # nuc_mask: np.ndarray, #
        # chain_token: np.ndarray, #gives each chain a different token 0-number of chains in datum
        # atom_token: np.ndarray, #shape of (len nucs, max_DNA)
        # atom_coord: np.ndarray, #shape of (len nucs, max_DNA, 3)
        # atom_mask
##PLOTTING:
    coords = dna_datum.atom_coord
    import plotly.graph_objects as go
    atom_names = np.array(all_atoms)[dna_datum.atom_token.astype(int)].reshape(-1) #all_atoms[dna_datum.atom_token]
    print(dna_datum.nuc_token.shape)
    x, y, z = coords.reshape(-1, 3).T
    print(dna_datum.chain_token)
    color_mapping = {
        0: 'rgb(31, 119, 180)',    # blue
        1: 'rgb(255, 127, 14)',    # orange
        2: 'rgb(44, 160, 44)',     # green
        3: 'rgb(214, 39, 40)',     # red
        4: 'rgb(148, 103, 189)',   # purple
        5: 'rgb(247, 182, 210)' ,  # light pink    
        6: 'rgb(227, 119, 194)',   # pink
        7: 'rgb(127, 127, 127)',   # gray
        8: 'rgb(188, 189, 34)',    # yellow
        9: 'rgb(23, 190, 207)',    # cyan
        10: 'rgb(174, 199, 232)',  # light blue
        11: 'rgb(255, 152, 150)',  # light red
        12: 'rgb(197, 176, 213)',  # light purple
        13: 'rgb(196, 156, 148)',  # light brown
    }
    colors = [color_mapping[token] for token in dna_datum.nuc_token for _ in range(24)]
    fig = go.Figure(data=[go.Scatter3d(mode='markers',
            x=x,
            y=y,
            z=z,
            text=atom_names,
            hovertemplate='<b>%{text}</b>',
            marker=dict(size=3),
            # color = dna_datum.nuc_token,
            line=dict(
                color=colors,
                width=3,)
                )])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=800,
        title='3D Scatter Plot of Atom Coordinates'
    )
    
    ####TO PLOT:
    # fig.show()

    def get_repetition_lengths(lst):
        lengths = []
        count = 1
        for i in range(1, len(lst)):
            if lst[i] == lst[i-1]:
                count += 1
            else:
                lengths.append(count)
                count = 1
        lengths.append(count)
        return lengths

    rna_res_names = ['A', 'U', 'RT', 'G', 'C', 'I']
    dna_res_names = ['DA', 'DU', 'DT', 'DG', 'DC', 'DI']
    dna_res_tokens = list(map(lambda res: get_nucleotide_index(res), dna_res_names))
    rna_res_tokens = list(map(lambda res: get_nucleotide_index(res), rna_res_names))

    rna_res_tokens_dict = {res: get_nucleotide_index(res) for res in rna_res_names}
    dna_res_tokens_dict = {res: get_nucleotide_index(res) for res in dna_res_names}

    
    num_chains = dna_datum.chain_token[-1]
    chains_len = get_repetition_lengths(dna_datum.chain_token)
    follow=[]
    for nuci in dna_datum.nuc_token:
        if nuci in rna_res_tokens:
            follow.append('R')
        elif nuci in dna_res_tokens:
            follow.append('D')
        else:
            follow.append('confused')
    print(dna_datum.nuc_token[140])
    print(len(dna_datum))
    print(len(dna_datum.nuc_token))
    #number of RNA chains, 
    # for chain in range(num_chains+1):
    #     chain_len = chains_len[chain]
        
    #number of DNA chains, (check for chain and RNA/DNA),

    # how long the chain is
    #total number of nucleotides, 
    
    breakpoint()
    



