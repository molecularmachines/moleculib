import numpy as np
from biotite.database import rcsb
from biotite.sequence import ProteinSequence
from biotite.structure import (
    apply_chain_wise,
    apply_residue_wise,
    get_chain_count,
    get_residue_count,
    get_residues,
    spread_chain_wise,
    spread_residue_wise,
)

from biotite.structure import filter_amino_acids

import biotite.structure.io.mmtf as mmtf

from .alphabet import (
    all_atoms,
    all_residues,
    backbone_atoms,
    atom_index,
    atom_to_residues_index,
    get_residue_index,
)


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
        **kwargs,
    ):
        self.idcode = idcode
        self.resolution = resolution
        self.sequence = str(sequence)
        self.residue_token = residue_token
        self.residue_index = residue_index
        self.residue_mask = residue_mask
        self.chain_token = chain_token
        self.atom_token = atom_token
        self.atom_coord = atom_coord
        self.atom_mask = atom_mask
        for key, value in kwargs.items():
            setattr(self, key, value)

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
            atom_array_ = atom_array[(atom_array.atom_name == atom_name)]
            # kill pads and kill unks that are not backbone
            atom_array_ = atom_array_[(atom_array_.residue_token > 0)]
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
        mmtf_file = mmtf.MMTFFile.read(filepath)
        # Note(Allan): come back here, remove model=1 and set dynamically
        atom_array = mmtf.get_structure(mmtf_file, model=1)
        header = dict(
            idcode=mmtf_file["structureId"],
            resolution=None
            if ("resolution" not in mmtf_file)
            else mmtf_file["resolution"],
        )
        aa_filter = filter_amino_acids(atom_array)
        atom_array = atom_array[aa_filter]
        return cls.from_atom_array(atom_array, header=header)

    @classmethod
    def fetch_pdb_id(cls, id, save_path=None):
        filepath = rcsb.fetch(id, "mmtf", save_path)
        return cls.from_filepath(filepath)

    @classmethod
    def from_atom_array(
        cls,
        atom_array,
        header,
        query_atoms=all_atoms,
    ):
        if atom_array.array_length() == 0:
            return cls.empty_protein()

        _, res_names = get_residues(atom_array)
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

        residue_mask = residue_mask & (atom_extract["atom_coord"].sum((-1, -2)) != 0)

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

    def to_dict(self, attrs=None):
        if attrs is None:
            attrs = vars(self).keys()
        dict_ = {}
        for attr in attrs:
            obj = getattr(self, attr)
            # strings are not JAX types
            if type(obj) == str:
                continue
            if type(obj) in [list, tuple]:
                if type(obj[0]) not in [int, float]:
                    continue
                obj = np.array(obj)
            dict_[attr] = obj
        return dict_
