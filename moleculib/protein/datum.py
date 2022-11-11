import os
from pathlib import Path
import numpy as np
from biotite.database import rcsb
from biotite.structure.io.npz import NpzFile
from biotite.sequence import ProteinSequence
from biotite.structure import (
    get_residues,
    get_residue_count,
    spread_residue_wise,
    apply_residue_wise,
    spread_chain_wise,
    apply_chain_wise,
    get_chain_count,
)
from .alphabet import (
    atom_index,
    residue_index,
    all_atoms,
    atom_to_residues_index,
)
from .utils import pdb_to_atom_array, config


class ProteinDatum:
    """
    Incorporates protein sequence data to MolecularDatum
    and reshapes atom arrays to residue-based representation
    """

    def __init__(
        self,
        pid: str,
        resolution: float,
        sequence: ProteinSequence,
        residue_token: np.ndarray,
        residue_mask: np.ndarray,
        chain_token: np.ndarray,
        atom_token: np.ndarray,
        atom_coord: np.ndarray,
        atom_mask: np.ndarray,
    ):
        self.pid = pid
        self.resolution = resolution
        self.sequence = sequence
        self.residue_token = residue_token
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

        def _atom_slice(atom_array, atom):
            atom_array_ = atom_array[(atom_array.atom_name == atom_name)]
            res_tokens, seq_id = atom_array_.residue_token, atom_array_.seq_uid
            atom_indices = atom_to_residues_index[atom_token][res_tokens]
            for attr in attrs:
                attr_tensor = getattr(atom_array_, attr)
                extraction[attr][seq_id, atom_indices, ...] = attr_tensor
            mask[seq_id, atom_indices] = True

        for atom_name in all_atoms:
            atom_token = all_atoms.index(atom_name)
            _atom_slice(atom_array, atom_token)

        return extraction, mask

    def __len__(self):
        return len(self.sequence)

    @classmethod
    def empty_protein(cls):
        return cls(
            pid="",
            resolution=0.0,
            sequence=ProteinSequence(""),
            residue_token=np.array([]),
            residue_mask=np.array([]),
            chain_token=np.array([]),
            atom_token=np.array([]),
            atom_coord=np.array([]),
            atom_mask=np.array([]),
        )

    @classmethod
    def from_filepath(cls, filepath, format="npz"):
        if format == "pdb":
            atom_array = pdb_to_atom_array(filepath)
        elif format == "npz":
            file = NpzFile.read(str(filepath))
            atom_array = file.get_structure()
        else:
            raise ValueError("format needs to be npz or pdb")
        pdb_id = filepath[-7:-3]
        header = dict(pid=pdb_id)

        return cls.from_atom_array(atom_array, header)

    @classmethod
    def fetch_from_pdb(cls, id, save_path=None, format="npz"):
        if (save_path is not None) and (format == "pdb"):
            pdb_save_path = save_path
        else:
            pdb_save_path = config["cache_dir"]

        filepath = rcsb.fetch(id, "pdb", pdb_save_path, verbose=True)
        atom_array = pdb_to_atom_array(filepath)

        if (save_path is not None) and (format == "npz"):
            save_path = Path(save_path)
            data_path = save_path / f"{id}.npz"
            os.makedirs(str(save_path), exist_ok=True)

            npz_file = NpzFile()
            npz_file.set_structure(atom_array)
            npz_file.write(str(data_path))

        header = dict(pid=id)
        return cls.from_atom_array(atom_array, header=header)

    @classmethod
    def from_atom_array(cls, atom_array, header, query_atoms=all_atoms):
        if atom_array.array_length() == 0:
            return cls.empty_protein()

        res_ids, res_names = get_residues(atom_array)
        sequence = ProteinSequence(res_names)

        atom_array.add_annotation("seq_uid", int)
        atom_array.seq_uid = spread_residue_wise(
            atom_array, np.arange(0, get_residue_count(atom_array))
        )

        atom_array.add_annotation("token", int)
        atom_array.token = np.array(
            list(map(lambda atom: atom_index(atom), atom_array.atom_name))
        )

        residue_token = np.array(
            list(map(lambda res: residue_index(res), atom_array.res_name))
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
        residue_mask = _reshape_residue_attr(residue_mask)
        chain_token = _reshape_residue_attr(chain_token)

        atom_extract = dict(
            map(lambda kv: (f"atom_{kv[0]}", kv[1]), atom_extract.items())
        )

        # NOTE(Allan): the easy fix for resolution requires modifying
        # biotite. It's an easy modification with PR to two different repos
        # I'll come back to this at some point.
        return cls(
            pid=header["pid"],
            sequence=sequence,
            resolution=0.0,
            residue_token=residue_token,
            residue_mask=residue_mask,
            chain_token=chain_token,
            **atom_extract,
            atom_mask=atom_mask,
        )
