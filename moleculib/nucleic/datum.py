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

from alphabet import (
    all_atoms,
    all_nucs,
    # backbone_atoms,
    atom_index,
    atom_to_nucs_index,
    get_nucleotide_index,
    MAX_DNA_ATOMS
)

#NOTE: fix it to come from alphabet
#NOTE: extra ' for carbons in sugar backbone
backbone_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'","P", "O1P", "O2P","O3P","O2'", "O3'", "O4'", "O5'"] 

from utils import  pdb_to_atom_array

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
        return len(self.sequence)

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
        for attr in attrs:
            attr_shape = getattr(atom_array, attr).shape
            if len(attr_shape) == 1:
                attr_reshape = np.zeros((residue_count, MAX_DNA_ATOMS))
            else:
                attr_reshape = np.zeros((residue_count, MAX_DNA_ATOMS, attr_shape[-1]))
            extraction[attr] = attr_reshape

        
        def _atom_slice(atom_name, atom_array, atom_token):
            atom_array_ = atom_array[(atom_array.atom_name == atom_name)]
            # kill pads and kill unks that are not backbone
            atom_array_ = atom_array_[(atom_array_.residue_token > 0)]
            
            # NOTE(Dana): this will throw an error eventually
            if atom_name not in backbone_atoms:
                atom_array_ = atom_array_[(atom_array_.residue_token > 1)]

            res_tokens, seq_id = atom_array_.residue_token, atom_array_.seq_uid
            atom_indices = atom_to_indices[atom_token][res_tokens]
            # if atom_name == "C8":
            #     breakpoint()    
            for attr in attrs:
                attr_tensor = getattr(atom_array_, attr)
                extraction[attr][seq_id, atom_indices, ...] = attr_tensor
            mask[seq_id, atom_indices] = True

        for atom_name in atom_alphabet:
            atom_token = atom_alphabet.index(atom_name)
            _atom_slice(atom_name, atom_array, atom_token)

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
        #from Ido's code: (not sure necassary)
        # idcode = os.path.basename(filepath)
        # idcode = os.path.splitext(idcode)[0]
        # header['idcode'] = idcode
        
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

if __name__ == '__main__':
    dna_datum = NucleicDatum.fetch_pdb_id('5F9R')
    coords = dna_datum.atom_coord
    
    # fig = _scatter_coord('59fr', coords, color='black', visible=True)
    import plotly.graph_objects as go
    x, y, z = coords.reshape(-1, 3).T
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
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

    fig.show()
    # 
    
    
    



