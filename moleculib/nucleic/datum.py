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
    all_residues,
    backbone_atoms,
    atom_index,
    atom_to_residues_index,
    get_residue_index,
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
        residue_token: np.ndarray,
        residue_index: np.ndarray,
        residue_mask: np.ndarray,
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
    def _extract_reshaped_atom_attr(cls, atom_array, attrs):
        chain_count = get_chain_count(atom_array)
        extraction = dict()
        mask= np.zeros((chain_count, 14)).astype(bool) #why 14?
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

#checking git

    
    



