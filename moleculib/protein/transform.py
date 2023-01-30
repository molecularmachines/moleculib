from .datum import ProteinDatum
from .alphabet import backbone_atoms, bonds_arr, bonds_mask
import numpy as np
from einops import rearrange


class ProteinTransform:
    """
    Abstract class for transformation of ProteinDatum datapoints
    """

    def transform(self, datum: ProteinDatum) -> ProteinDatum:
        """
        Takes as input an individual data point, processes
        the values in it and returns a new ProteinDatum
        """
        raise NotImplementedError("method transform must be implemented")


class ProteinCrop(ProteinTransform):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def transform(self, datum):
        return ProteinDatum(
            idcode=datum.idcode,
            resolution=datum.resolution,
            residue_index=datum.residue_index[: self.crop_size],
            sequence=datum.sequence[: self.crop_size],
            residue_token=datum.residue_token[: self.crop_size],
            residue_mask=datum.residue_mask[: self.crop_size],
            chain_token=datum.chain_token[: self.crop_size],
            atom_token=datum.atom_token[: self.crop_size],
            atom_coord=datum.atom_coord[: self.crop_size],
            atom_mask=datum.atom_mask[: self.crop_size],
        )


class ListBonds(ProteinTransform):
    """
    Augments ProteinDatum with bonds_list, a list of atomic indexes connected by a bond
    Note that indexing is performed at atom level, that is, as residue_dim atom_dim -> (residue_dim atom_dim)
    """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size

    def transform(self, datum):
        # solve for intra-residue bonds
        num_atoms = datum.atom_coord.shape[-2]
        count = num_atoms * np.expand_dims(
            np.arange(0, len(datum.residue_token)), axis=(-1, -2)
        )
        bonds_per_residue = bonds_arr[datum.residue_token]

        bonds_mask_per_residue = bonds_mask[datum.residue_token].squeeze(-1)
        bonds_list = (bonds_per_residue + count)[bonds_mask_per_residue].astype(
            np.int32
        )

        # add peptide bonds
        ns = num_atoms * np.arange(1, len(datum.atom_coord)) + backbone_atoms.index("N")
        cs = num_atoms * np.arange(0, len(datum.atom_coord) - 1) + backbone_atoms.index(
            "C"
        )
        peptide_bonds = np.stack((ns, cs)).T
        bonds_list = np.concatenate((bonds_list, peptide_bonds), axis=0)

        # kill bonds for which atom_mask flags lack record
        left, right = bonds_list.T
        atom_mask = rearrange(datum.atom_mask, "r a -> (r a)")
        bonds_list = bonds_list[atom_mask[left] & atom_mask[right]]

        # used fixed-size buffer for sake of jit-friendliness
        buffer = np.zeros((self.buffer_size, 2), dtype=np.int32)
        buffer_mask = np.zeros((self.buffer_size,)).astype(np.bool_)

        buffer[: len(bonds_list), :] = bonds_list
        buffer_mask[: len(bonds_list)] = True

        datum.bonds_list = buffer
        datum.bonds_mask = buffer_mask

        return datum


class MaybeMirror(ProteinTransform):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def transform(self, datum):
        return datum
