from .datum import ProteinDatum
from .alphabet import (
    backbone_atoms,
    bonds_arr,
    bonds_mask,
    angles_arr,
    angles_mask,
    dihedrals_arr,
    dihedrals_mask,
)
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


class ListAngles(ProteinTransform):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size

    def transform(self, datum):
        # solve for intra-residue angles
        num_atoms = datum.atom_coord.shape[-2]
        count = num_atoms * np.expand_dims(
            np.arange(0, len(datum.residue_token)), axis=(-1, -2)
        )
        angles_per_residue = angles_arr[datum.residue_token]

        angles_mask_per_residue = angles_mask[datum.residue_token].squeeze(-1)
        angles_list = (angles_per_residue + count)[angles_mask_per_residue].astype(
            np.int32
        )

        # solve for inter-residue angles
        # ["prev CA", "prev C", "next N"]
        # ["prev C", "next N", "next CA"]
        # add peptide bonds
        def prev_(atom):
            page = num_atoms * np.arange(0, len(datum.atom_coord) - 1)
            return page + backbone_atoms.index(atom)

        def next_(atom):
            page = num_atoms * np.arange(1, len(datum.atom_coord))
            return page + backbone_atoms.index(atom)

        first_angle = np.stack((prev_("CA"), prev_("C"), next_("N"))).T
        second_angle = np.stack((prev_("C"), next_("N"), next_("CA"))).T
        angles_list = np.concatenate((angles_list, first_angle, second_angle), axis=0)

        # kill bonds for which atom_mask flags lack record
        i, j, k = angles_list.T
        atom_mask = rearrange(datum.atom_mask, "r a -> (r a)")
        angles_list = angles_list[atom_mask[i] & atom_mask[j] & atom_mask[k]]

        # used fixed-size buffer for sake of jit-friendliness
        buffer = np.zeros((self.buffer_size, 3), dtype=np.int32)
        buffer_mask = np.zeros((self.buffer_size,)).astype(np.bool_)

        buffer[: len(angles_list), :] = angles_list
        buffer_mask[: len(angles_list)] = True

        datum.angles_list = buffer
        datum.angles_mask = buffer_mask

        return datum


class ListDihedrals(ProteinTransform):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size

    def transform(self, datum):
        # solve for intra-residue angles
        num_atoms = datum.atom_coord.shape[-2]
        count = num_atoms * np.expand_dims(
            np.arange(0, len(datum.residue_token)), axis=(-1, -2)
        )
        dihedrals_per_residue = dihedrals_arr[datum.residue_token]

        dihedrals_mask_per_residue = dihedrals_mask[datum.residue_token].squeeze(-1)
        dihedrals_list = dihedrals_per_residue + count
        dihedrals_list = dihedrals_list[dihedrals_mask_per_residue]
        dihedrals_list = dihedrals_list.astype(np.int32)

        # solve for inter-residue dihedrals
        # TODO(Allan): Not sure if we should include psi, phi and omega here

        # kill bonds for which atom_mask flags lack record
        p, q, u, v = dihedrals_list.T
        atom_mask = rearrange(datum.atom_mask, "r a -> (r a)")
        angles_list = dihedrals_list[
            atom_mask[p] & atom_mask[q] & atom_mask[u] & atom_mask[v]
        ]

        # used fixed-size buffer for sake of jit-friendliness
        buffer = np.zeros((self.buffer_size, 4), dtype=np.int32)
        buffer_mask = np.zeros((self.buffer_size,)).astype(np.bool_)

        buffer[: len(angles_list), :] = angles_list
        buffer_mask[: len(angles_list)] = True

        datum.dihedrals_list = buffer
        datum.dihedrals_mask = buffer_mask

        return datum


class DescribeChemistry(ProteinTransform):
    """
    Augments ProteinDatum with bonds_list and angles_list a list of atomic indexes connected by bonds and angles
    that should have invariant behavior according to chemistry
    Note that indexing is performed at atom level, that is, as residue_dim atom_dim -> (residue_dim atom_dim)
    """

    def __init__(self, bond_buffer_size, angle_buffer_size, dihedral_buffer_size):
        self.bond_transform = ListBonds(bond_buffer_size)
        self.angle_transform = ListAngles(angle_buffer_size)
        self.dihedral_transform = ListDihedrals(dihedral_buffer_size)

    def transform(self, datum):
        datum = self.bond_transform.transform(datum)
        datum = self.angle_transform.transform(datum)
        datum = self.dihedral_transform.transform(datum)
        return datum


class MaybeMirror(ProteinTransform):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def transform(self, datum):
        return datum
