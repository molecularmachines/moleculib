import numpy as np
from Bio.PDB import parse_pdb_header
from biotite.database import rcsb
from biotite.sequence import ProteinSequence, NucleotideSequence
from biotite.structure import (
    apply_chain_wise,
    apply_residue_wise,
    get_chain_count,
    get_residue_count,
    get_residues,
    spread_chain_wise,
    spread_residue_wise,
)
from biotite.structure import filter_nucleotides

import biotite.structure.io.mmtf as mmtf

from .alphabet import (
    all_atoms,
    all_nucs,
    backbone_atoms,
    atom_index,
    atom_to_nucs_index,
    get_nucleotide_index,
)

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
    
    @classmethod
    def _extract_reshaped_atom_attr(cls, atom_array, attrs):
        residue_count = get_residue_count(atom_array)
        extraction = dict()
        mask = np.zeros((residue_count, 14)).astype(bool)


    def __len__(self):
        return len(self.sequence)

    @classmethod
    def _extract_reshaped_atom_attr(cls, atom_array, attrs):
        """
        Given the alphabet, it will extract all atoms of a residue in the alphabets order
        if theres more atoms than the largest alphabet size, it will pad
        """
        chain_count = get_chain_count(atom_array) #numbrer of chains
        extraction = dict()
        mask= np.zeros((chain_count, 14)).astype(bool) #why 14? # array of Falses
        for attr in attrs:
            attr_shape = getattr(atom_array, attr).shape 
            if len(attr_shape) == 1:
                attr_reshape = np.zeros((chain_count, 14))
            else:
                attr_reshape = np.zeros((chain_count, 14, attr_shape[-1]))
            extraction[attr] = attr_reshape

        def _atom_slice(atom_name, atom_array, atom_token):
            atom_array_ = atom_array[(atom_array.atom_name == atom_name)]
            # kill pads and kill unks that are not backbone
            atom_array_ = atom_array_[(atom_array_.nuc_token > 0)]
            if atom_name not in backbone_atoms:
                atom_array_ = atom_array_[(atom_array_.nuc_token > 1)]

            nuc_tokens, seq_id = atom_array_.nuc_token, atom_array_.seq_uid
            ###TODO change atom to nuc in alphabet
            atom_indices = atom_to_nucs_index[atom_token][nuc_tokens]
            for attr in attrs:
                attr_tensor = getattr(atom_array_, attr)
                extraction[attr][seq_id, atom_indices, ...] = attr_tensor # what ... does?
            mask[seq_id, atom_indices] = True

        for atom_name in all_atoms:
            atom_token = all_atoms.index(atom_name)
            _atom_slice(atom_name, atom_array, atom_token)

        return extraction, mask
            

    @classmethod
    def empty_nuc(cls):
        return cls(
            idcode="",
            resolution=0.0,
            sequence=ProteinSequence(""),
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
    def from_atom_array(cls, atom_array, header, query_atoms=all_atoms):
        """
        creates a NucleicDatum from atom array
        """
        if atom_array.array_length() == 0:
            return cls.empty_nuc()
        #### TODO #####
        # return cls(
            # idcode="",
            # resolution=0.0,
            # sequence=ProteinSequence(""),
            # nuc_index=np.array([]),
            # nuc_token=np.array([]),
            # nuc_mask=np.array([]),
            # chain_token=np.array([]),
            # atom_token=np.array([]),
            # atom_coord=np.array([]),
            # atom_mask=np.array([])
            # ) i


import plotly.graph_objects as go
def _scatter(name, ca_coord, atom_coord, atom_mask, color, visible=True):
    sc_coords = []
    for ca, atoms, mask in zip(ca_coord, atom_coord, atom_mask):
        for atom in atoms[mask]:
            sc_coords.append(ca)
            sc_coords.append(atom)
            sc_coords.append([None, None, None])

    sc_coords = np.array(sc_coords)
    bb_x, bb_y, bb_z = ca_coord.T
    sc_x, sc_y, sc_z = sc_coords.T

    data = [go.Scatter3d(
            name=name + " coord",
            x=bb_x,
            y=bb_y,
            z=bb_z,
            marker=dict(
                size=7,
                colorscale="Viridis",
            ),
            line=dict(color=color, width=4),
            visible="legendonly" if not visible else True,
        ),
        go.Scatter3d(
            name=name + " vecs",
            x=sc_x,
            y=sc_y,
            z=sc_z,
            marker=dict(size=2, colorscale="Viridis"),
            line=dict(
                color=color,
                width=2,
            ),
            visible="legendonly",
        )]
    return data

if __name__ == '__main__':
    dna_datum = NucleicDatum.fetch_pdb_id('5F9R')
    
    
    



