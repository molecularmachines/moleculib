import os
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

from .alphabet import (
    all_atoms,
    all_residues,
    backbone_atoms,
    atom_index,
    atom_to_residues_index,
    get_residue_index,
    get_nucleotide_index,
    MAX_DNA_ATOMS,
)
from .utils import pdb_to_atom_array


class ProteinDatum:
    """
    Incorporates protein sequence data to MolecularDatum
    and reshapes atom arrays to residue-based representation
    """

    def __init__(
        self,
        idcode: str,
        resolution: float,
        sequence: ProteinSequence,
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
        self.sequence = sequence
        self.residue_token = residue_token
        self.residue_index = residue_index
        self.residue_mask = residue_mask
        self.chain_token = chain_token
        self.atom_token = atom_token
        self.atom_coord = atom_coord
        self.atom_mask = atom_mask

    @classmethod
    def _extract_reshaped_atom_attr(cls, atom_array, attrs):
        residue_count = get_residue_count(atom_array)

        extraction = dict()
        mask = np.zeros((residue_count, 14)).astype(bool)
        for attr in attrs:
            attr_shape = getattr(atom_array, attr).shape
            if len(attr_shape) == 1:
                attr_reshape = np.zeros((residue_count, 14))
            else:
                attr_reshape = np.zeros((residue_count, 14, attr_shape[-1]))
            extraction[attr] = attr_reshape

        def _atom_slice(atom_name, atom_array, atom_token):
            atom_array_ = atom_array[
                (atom_array.atom_name == atom_name) 
            ]
            if atom_name not in backbone_atoms:
                atom_array_ = atom_array_[(atom_array_.residue_token > 1)]

            res_tokens, seq_id = atom_array_.residue_token, atom_array_.seq_uid
            atom_indices = atom_to_residues_index[atom_token][res_tokens]
            for attr in attrs:
                attr_tensor = getattr(atom_array_, attr)
                extraction[attr][seq_id, atom_indices, ...] = attr_tensor
            mask[seq_id, atom_indices] = True

        for atom_name in all_atoms:
            atom_token = all_atoms.index(atom_name)
            _atom_slice(atom_name, atom_array, atom_token)

        return extraction, mask

    def __len__(self):
        return len(self.sequence)

    @classmethod
    def empty_protein(cls):
        return cls(
            idcode="",
            resolution=0.0,
            sequence=ProteinSequence(""),
            residue_index=np.array([]),
            residue_token=np.array([]),
            residue_mask=np.array([]),
            chain_token=np.array([]),
            atom_token=np.array([]),
            atom_coord=np.array([]),
            atom_mask=np.array([]),
        )

    @classmethod
    def from_filepath(cls, filepath):
        atom_array = pdb_to_atom_array(filepath)
        header = parse_pdb_header(filepath)
        idcode = os.path.basename(filepath)
        idcode = os.path.splitext(idcode)[0]
        header['idcode'] = idcode
        return cls.from_atom_array(atom_array, header=header)

    @classmethod
    def fetch_pdb_id(cls, id, save_path=None):
        filepath = rcsb.fetch(id, "pdb", save_path)
        return cls.from_filepath(filepath)

    @classmethod
    def from_atom_array(cls, atom_array, header, query_atoms=all_atoms):
        if atom_array.array_length() == 0:
            return cls.empty_protein()

        res_ids, res_names = get_residues(atom_array)
        res_names = [
            ("UNK" if (name not in all_residues) else name) for name in res_names
        ]
        sequence = ProteinSequence(list(res_names))

        atom_array.add_annotation("seq_uid", int)
        atom_array.seq_uid = spread_residue_wise(
            atom_array, np.arange(0, get_residue_count(atom_array))
        )

        atom_array.add_annotation("token", int)
        atom_array.token = np.array(
            list(map(lambda atom: atom_index(atom), atom_array.atom_name))
        )

        residue_token = np.array(
            list(map(lambda res: get_residue_index(res), atom_array.res_name))
        )
        residue_mask = np.ones_like(residue_token).astype(bool)

        atom_array.add_annotation("residue_token", int)
        atom_array.residue_token = residue_token
        chain_token = spread_chain_wise(
            atom_array, np.arange(0, get_chain_count(atom_array))
        )

        atom_array.add_annotation("res_uid", int)

        def _count_residues_per_chain(chain_atom_array, axis=0):
            return get_residue_count(chain_atom_array)

        chain_res_sizes = apply_chain_wise(
            atom_array, atom_array, _count_residues_per_chain, axis=0
        )
        chain_res_cumsum = np.cumsum([0] + list(chain_res_sizes[:-1]))
        atom_array.res_uid = atom_array.res_id + chain_res_cumsum[chain_token]

        atom_extract, atom_mask = cls._extract_reshaped_atom_attr(
            atom_array, ["coord", "token"]
        )

        def _pool_residue_token(atom_residue_tokens, axis=0):
            representative = atom_residue_tokens[0]
            return representative

        def _reshape_residue_attr(attr):
            return apply_residue_wise(atom_array, attr, _pool_residue_token, axis=0)

        residue_token = _reshape_residue_attr(residue_token)
        residue_index = np.arange(0, residue_token.shape[0])
        residue_mask = _reshape_residue_attr(residue_mask)
        chain_token = _reshape_residue_attr(chain_token)

        atom_extract = dict(
            map(lambda kv: (f"atom_{kv[0]}", kv[1]), atom_extract.items())
        )

        return cls(
            idcode=header["idcode"],
            sequence=sequence,
            resolution=header["resolution"],
            residue_token=residue_token,
            residue_index=residue_index,
            residue_mask=residue_mask,
            chain_token=chain_token,
            **atom_extract,
            atom_mask=atom_mask,
        )


class ProteinDNADatum(ProteinDatum):
    """
    Data class for proteins and DNA complexes
    """

    def __init__(
        self,
        idcode: str,
        resolution: float,
        sequence: ProteinSequence,
        residue_token: np.ndarray,
        residue_index: np.ndarray,
        residue_mask: np.ndarray,
        chain_token: np.ndarray,
        atom_token: np.ndarray,
        atom_coord: np.ndarray,
        atom_mask: np.ndarray,
        dna_sequence: NucleotideSequence,
        dna_token: np.ndarray,
        dna_coord: np.ndarray,
        dna_mask: np.ndarray,
    ):
        super().__init__(
            idcode=idcode,
            resolution=resolution,
            sequence=sequence,
            residue_token=residue_token,
            residue_index=residue_index,
            residue_mask=residue_mask,
            chain_token=chain_token,
            atom_token=atom_token,
            atom_coord=atom_coord,
            atom_mask=atom_mask
        )

        self.dna_sequence = dna_sequence
        self.dna_token = dna_token
        self.dna_coord = dna_coord
        self.dna_mask = dna_mask

    @classmethod
    def from_filepath(cls, filepath):
        dna_array = pdb_to_atom_array(filepath, dna=True)
        res_array = pdb_to_atom_array(filepath, dna=False)
        header = parse_pdb_header(filepath)
        return cls.from_atom_arrays(res_array, dna_array, header=header)

    @classmethod
    def from_atom_arrays(cls, res_array, dna_array, header, query_atoms=all_atoms):
        # residues are the same as ProteinDatum
        p = ProteinDatum.from_atom_array(res_array, header, query_atoms)

        # support proteins without DNA atoms
        if not len(dna_array):
            dna_sequence = NucleotideSequence("")
            dna_token, dna_coord, dna_mask = None, None, None

        else:
            # retrieve data from dna atom array
            _, nuc_names = get_residues(dna_array)
            nuc_names = [n[1] for n in nuc_names]  # in PDB nucleotides start with D
            dna_sequence = NucleotideSequence("".join(nuc_names))
            dna_token = np.array([get_nucleotide_index(n) for n in nuc_names])

            # identify individual nucleotide atoms from array
            nuc_atom_indices = []
            curr_nuc_id = dna_array[0].res_id
            curr_chain_id = dna_array[0].chain_id
            num_atoms = len(dna_array)
            for i in range(num_atoms):
                atom = dna_array[i]
                if atom.res_id != curr_nuc_id or atom.chain_id != curr_chain_id:
                    nuc_atom_indices.append(i)
                    curr_nuc_id = atom.res_id
                    curr_chain_id = atom.chain_id
            nuc_atom_indices.append(num_atoms)

            # retrieve atoms per nucleotide and masks
            dna_coord = np.zeros((len(dna_sequence), MAX_DNA_ATOMS, 3))
            dna_mask = np.zeros((len(dna_sequence), MAX_DNA_ATOMS))
            prev_idx = 0
            for i, idx in enumerate(nuc_atom_indices):
                # in case nucleotide has more than max atoms, trim last atoms
                num_nuc_atoms = min(idx - prev_idx, MAX_DNA_ATOMS)
                max_idx = min(idx, prev_idx + MAX_DNA_ATOMS)
                curr_nuc_atoms = dna_array.coord[prev_idx:max_idx]
                dna_coord[i][:num_nuc_atoms] = curr_nuc_atoms
                dna_mask[i][:num_nuc_atoms] = 1
                prev_idx = idx
            dna_mask = dna_mask.astype(bool)

        # construct protein dna complex
        return cls(
            idcode=p.idcode,
            sequence=p.sequence,
            resolution=p.resolution,
            residue_token=p.residue_token,
            residue_index=p.residue_index,
            residue_mask=p.residue_mask,
            chain_token=p.chain_token,
            atom_coord=p.atom_coord,
            atom_token=p.atom_token,
            atom_mask=p.atom_mask,
            dna_sequence=dna_sequence,
            dna_token=dna_token,
            dna_coord=dna_coord,
            dna_mask=dna_mask,
        )
