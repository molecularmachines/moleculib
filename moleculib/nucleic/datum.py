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
from einops import rearrange, repeat


import plotly.graph_objects as go
import plotly.offline as pyo

import sys
sys.path.append('.')

from .alphabet import *



# from utils import  pdb_to_atom_array
import os
from pathlib import Path

from biotite.structure import filter_nucleotides
from biotite.structure.io.pdb import PDBFile


import numpy as np

home_dir = str(Path.home()) #not sure what this do
config = {"cache_dir": os.path.join(home_dir, ".cache", "moleculib")} #not sure either


def pdb_to_atom_array(pdb_path, RNA=True):
    pdb_file = PDBFile.read(pdb_path)
    atom_array = pdb_file.get_structure(
        model=1, extra_fields=["atom_id", "b_factor", "occupancy", "charge"])
    nuc_filter = filter_nucleotides(atom_array)
    if RNA==True:
        DNA = ["DA", "DC", "DG", "DI", "DT", "DU"]
        dna_filter = np.isin(atom_array.res_name, DNA) #filters for DNA only
        
        no_dna_filter = np.logical_not(dna_filter)
        # print(len(dna_filter),len(nuc_filter))
        RNA_filter = np.logical_and(no_dna_filter, nuc_filter) #no DNA and only nucleotides filter
        nuc_filter=RNA_filter

    atom_array = atom_array[nuc_filter]
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
        **kwargs, 
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
        for key, value in kwargs.items():
            setattr(self, key, value)
    

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
        atom_array =  pdb_to_atom_array(filepath, RNA=True) #filters pdb to only nucleotides
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
        # print("length of atom array: " , len(atom_array))
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
        # print("line 234 in datum, that is atom_array.res_name:", len(atom_array.res_name))
        # print("line 234 in datum, that is atom_array.res_name:", len(residue_token))
        
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


    def to_pdb_str(self):
        # https://colab.research.google.com/github/pb3lab/ibm3202/blob/master/tutorials/lab02_molviz.ipynb#scrollTo=FPS04wJf5k3f
        assert len(self.nuc_token.shape) == 1
        atom_mask = self.atom_mask.astype(np.bool_)
        all_atom_coords = self.atom_coord[atom_mask]
        all_atom_tokens = self.atom_token[atom_mask]
        all_atom_res_tokens = repeat(self.nuc_token, "r -> r a", a=24)[atom_mask]
        all_atom_res_indices = repeat(self.nuc_index, "r -> r a", a=24)[atom_mask]

        # just in case, move to cpu
        atom_mask = np.array(atom_mask)
        all_atom_coords = np.array(all_atom_coords)
        all_atom_tokens = np.array(all_atom_tokens)
        all_atom_res_tokens = np.array(all_atom_res_tokens) #all_nucs_atom_tokens
        all_atom_res_indices = np.array(all_atom_res_indices)

        lines = []
        for idx, (coord, token, res_token, res_index) in enumerate(
            zip(
                all_atom_coords,
                all_atom_tokens,
                all_atom_res_tokens,
                all_atom_res_indices,
            )
        ):
            name = all_atoms[int(token)]
            res_name = all_nucs[int(res_token)]
            x, y, z = coord
            line = list(" " * 80)
            line[0:6] = "ATOM".ljust(6)
            line[6:11] = str(idx + 1).ljust(5)
            line[12:16] = name.ljust(4)
            line[17:20] = res_name.ljust(3)
            line[21:22] = "A" ##if that is chain identifier it should go from a to z?
            line[23:27] = str(res_index + 1).ljust(4)
            line[30:38] = f"{x:.3f}".rjust(8)
            line[38:46] = f"{y:.3f}".rjust(8)
            line[46:54] = f"{z:.3f}".rjust(8)
            line[76:78] = name[0].rjust(2)
            lines.append("".join(line))
        lines = "\n".join(lines)
        return lines


    def plot(
        self, 
        view, 
        viewer=None, 
        sphere=False, 
        ribbon=True,
        sidechain=True,
        color='spectrum',
    ):
        if viewer is None:
            viewer = (0, 0)
        view.addModel(self.to_pdb_str(), 'pdb', viewer=viewer)
        view.setStyle({'model': -1}, {}, viewer=viewer)
        if sphere:
            view.addStyle({'model': -1}, {'sphere': {'radius': 0.3}}, viewer=viewer)

        if ribbon:
            view.addStyle({'model': -1}, {'cartoon': {'color': color}}, viewer=viewer)

        if sidechain:
            if color != 'spectrum':
                view.addStyle({'model': -1}, {'stick': {'radius': 0.2, 'color': color}}, viewer=viewer)
            else:
                view.addStyle({'model': -1}, {'stick': {'radius': 0.2}}, viewer=viewer)

        return view



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
