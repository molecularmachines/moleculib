from .datum import MoleculeDatum

# from .alphabet import (
#     all_atoms,
#     backbone_atoms,
#     bonds_arr,
#     bonds_mask,
#     angles_arr,
#     angles_mask,
#     dihedrals_arr,
#     dihedrals_mask,
#     flippable_arr,
#     flippable_mask,
# )
import numpy as np
from einops import rearrange
from .utils import pad_array
from functools import partial


class MoleculeTransform:
    """
    Abstract class for transformation of MoleculeDatum datapoints
    """

    def transform(self, datum: MoleculeDatum) -> MoleculeDatum:
        """
        Takes as input an individual data point, processes
        the values in it and returns a new MoleculeDatum
        """
        raise NotImplementedError("method transform must be implemented")


class MoleculeCrop(MoleculeTransform):
    def __init__(self, crop_size, pad=True):
        self.crop_size = crop_size
        if pad:
            self.padding = partial(pad_array, total_size=crop_size)

    def transform(self, datum):
        atom_count = len(datum.atom_token)
        if atom_count <= self.crop_size and hasattr(self, "padding"):
            new_datum = datum  # TODO change this to include important stuff
            new_datum.atom_token = self.padding(datum.atom_token)
            new_datum.atom_coord = self.padding(datum.atom_coord)
            new_datum.b_factor = self.padding(datum.b_factor)
        else:
            cut = np.random.randint(low=0, high=(atom_count - self.crop_size))
            new_datum = datum
            new_datum.atom_token = datum.atom_token[cut : cut + self.crop_size]
            new_datum.atom_coord = datum.atom_coord[cut : cut + self.crop_size]
            new_datum.b_factor = datum.b_factor[cut : cut + self.crop_size]
        new_datum.crop_size = self.crop_size
        return new_datum


# class ListBonds(MoleculeTransform):
#     def __init__(self, buffer_size):
#         self.buffer_size = buffer_size

#     def transform(self, datum):
#         # solve for intra-residue bonds
#         num_atoms = datum.atom_coord.shape[-2]
#         count = num_atoms * np.expand_dims(
#             np.arange(0, len(datum.residue_token)), axis=(-1, -2)
#         )
#         bonds_per_residue = bonds_arr[datum.residue_token]

#         bonds_mask_per_residue = bonds_mask[datum.residue_token].squeeze(-1)
#         bonds_list = (bonds_per_residue + count)[bonds_mask_per_residue].astype(
#             np.int32
#         )

#         # add peptide bonds
#         ns = num_atoms * np.arange(1, len(datum.atom_coord)) + backbone_atoms.index("N")
#         cs = num_atoms * np.arange(0, len(datum.atom_coord) - 1) + backbone_atoms.index(
#             "C"
#         )
#         peptide_bonds = np.stack((ns, cs)).T
#         bonds_list = np.concatenate((bonds_list, peptide_bonds), axis=0)

#         # kill bonds for which atom_mask flags lack record
#         left, right = bonds_list.T
#         atom_mask = rearrange(datum.atom_mask, "r a -> (r a)")
#         bonds_list = bonds_list[atom_mask[left] & atom_mask[right]]

#         # used fixed-size buffer for sake of jit-friendliness
#         buffer = np.zeros((self.buffer_size, 2), dtype=np.int32)
#         buffer_mask = np.zeros((self.buffer_size,)).astype(np.bool_)

#         buffer[: len(bonds_list), :] = bonds_list
#         buffer_mask[: len(bonds_list)] = True

#         datum.bonds_list = buffer
#         datum.bonds_mask = buffer_mask

#         return datum


# class ListAngles(MoleculeTransform):
#     def __init__(self, buffer_size):
#         self.buffer_size = buffer_size

#     def transform(self, datum):
#         # solve for intra-residue angles
#         num_atoms = datum.atom_coord.shape[-2]
#         count = num_atoms * np.expand_dims(
#             np.arange(0, len(datum.residue_token)), axis=(-1, -2)
#         )
#         angles_per_residue = angles_arr[datum.residue_token]

#         angles_mask_per_residue = angles_mask[datum.residue_token].squeeze(-1)
#         angles_list = (angles_per_residue + count)[angles_mask_per_residue].astype(
#             np.int32
#         )

#         # solve for inter-residue angles
#         # ["prev CA", "prev C", "next N"]
#         # ["prev C", "next N", "next CA"]
#         # add peptide bonds
#         def prev_(atom):
#             page = num_atoms * np.arange(0, len(datum.atom_coord) - 1)
#             return page + backbone_atoms.index(atom)

#         def next_(atom):
#             page = num_atoms * np.arange(1, len(datum.atom_coord))
#             return page + backbone_atoms.index(atom)

#         first_angle = np.stack((prev_("CA"), prev_("C"), next_("N"))).T
#         second_angle = np.stack((prev_("C"), next_("N"), next_("CA"))).T
#         angles_list = np.concatenate((angles_list, first_angle, second_angle), axis=0)

#         # kill bonds for which atom_mask flags lack record
#         i, j, k = angles_list.T
#         atom_mask = rearrange(datum.atom_mask, "r a -> (r a)")
#         angles_list = angles_list[atom_mask[i] & atom_mask[j] & atom_mask[k]]

#         # used fixed-size buffer for sake of jit-friendliness
#         buffer = np.zeros((self.buffer_size, 3), dtype=np.int32)
#         buffer_mask = np.zeros((self.buffer_size,)).astype(np.bool_)

#         buffer[: len(angles_list), :] = angles_list
#         buffer_mask[: len(angles_list)] = True

#         datum.angles_list = buffer
#         datum.angles_mask = buffer_mask

#         return datum


# class ListDihedrals(MoleculeTransform):
#     def __init__(self, buffer_size):
#         self.buffer_size = buffer_size

#     def transform(self, datum):
#         # solve for intra-residue angles
#         num_atoms = datum.atom_coord.shape[-2]
#         count = num_atoms * np.expand_dims(
#             np.arange(0, len(datum.residue_token)), axis=(-1, -2)
#         )
#         dihedrals_per_residue = dihedrals_arr[datum.residue_token]

#         dihedrals_mask_per_residue = dihedrals_mask[datum.residue_token].squeeze(-1)
#         dihedrals_list = dihedrals_per_residue + count
#         dihedrals_list = dihedrals_list[dihedrals_mask_per_residue]
#         dihedrals_list = dihedrals_list.astype(np.int32)

#         def prev_(atom):
#             page = num_atoms * np.arange(0, len(datum.atom_coord) - 1)
#             return page + all_atoms.index(atom)

#         def next_(atom):
#             page = num_atoms * np.arange(1, len(datum.atom_coord))
#             return page + all_atoms.index(atom)

#         psi = np.stack((prev_("N"), prev_("CA"), prev_("C"), next_("N"))).T
#         omega = np.stack((prev_("CA"), prev_("C"), next_("N"), next_("CA"))).T
#         phi = np.stack((prev_("C"), next_("N"), next_("CA"), next_("C"))).T
#         dihedrals_list = np.concatenate((dihedrals_list, psi, phi, omega), axis=0)

#         mirror_break = np.stack((prev_("C"), next_("N"), next_("CA"), next_("CB"))).T
#         mirror_break2 = np.stack((prev_("CB"), prev_("CA"), prev_("C"), next_("N"))).T
#         dihedrals_list = np.concatenate(
#             (dihedrals_list, mirror_break, mirror_break2), axis=0
#         )

#         # kill bonds for which atom_mask flags lack record
#         p, q, u, v = dihedrals_list.T
#         atom_mask = rearrange(datum.atom_mask, "r a -> (r a)")
#         dihedrals_list = dihedrals_list[
#             atom_mask[p] & atom_mask[q] & atom_mask[u] & atom_mask[v]
#         ]

#         # used fixed-size buffer for sake of jit-friendliness
#         buffer = np.zeros((self.buffer_size, 4), dtype=np.int32)
#         buffer_mask = np.zeros((self.buffer_size,)).astype(np.bool_)

#         buffer[: len(dihedrals_list), :] = dihedrals_list
#         buffer_mask[: len(dihedrals_list)] = True

#         datum.dihedrals_list = buffer
#         datum.dihedrals_mask = buffer_mask

#         return datum


# class ListMirrorFlips(MoleculeTransform):
#     def __init__(self, buffer_size):
#         self.buffer_size = buffer_size

#     def transform(self, datum):
#         # solve for intra-residue angles
#         num_atoms = datum.atom_coord.shape[-2]
#         count = num_atoms * np.expand_dims(
#             np.arange(0, len(datum.residue_token)), axis=(-1, -2)
#         )
#         flips_per_residue = flippable_arr[datum.residue_token]

#         flips_mask_per_residue = flippable_mask[datum.residue_token].squeeze(-1)
#         flips_list = (flips_per_residue + count)[flips_mask_per_residue].astype(
#             np.int32
#         )
#         # used fixed-size buffer for sake of jit-friendliness
#         buffer = np.zeros((self.buffer_size, 2), dtype=np.int32)
#         buffer_mask = np.zeros((self.buffer_size,)).astype(np.bool_)

#         buffer[: len(flips_list), :] = flips_list
#         buffer_mask[: len(flips_list)] = True

#         datum.flips_list = buffer
#         datum.flips_mask = buffer_mask

#         return datum


# class DescribeChemistry(MoleculeTransform):
#     """
#     Augments MoleculeDatum with bonds_list and angles_list a list of atomic indexes connected by bonds and angles
#     that should have invariant behavior according to chemistry
#     Note that indexing is performed at atom level, that is, as residue_dim atom_dim -> (residue_dim atom_dim)
#     """

#     def __init__(
#         self,
#         bond_buffer_size,
#         angle_buffer_size,
#         dihedral_buffer_size,
#         flips_buffer_size,
#     ):
#         self.bond_transform = ListBonds(bond_buffer_size)
#         self.angle_transform = ListAngles(angle_buffer_size)
#         self.dihedral_transform = ListDihedrals(dihedral_buffer_size)
#         self.flip_transform = ListMirrorFlips(flips_buffer_size)

#     def transform(self, datum):
#         datum = self.bond_transform.transform(datum)
#         datum = self.angle_transform.transform(datum)
#         datum = self.dihedral_transform.transform(datum)
#         datum = self.flip_transform.transform(datum)
#         return datum


# def normalize(vector):
#     norms_sqr = np.sum(vector**2, axis=-1, keepdims=True)
#     norms = np.where(norms_sqr == 0.0, 1.0, norms_sqr) ** 0.5
#     return vector / norms


# def measure_chirality(coords):
#     n = coords[:, backbone_atoms.index("N")]
#     ca = coords[:, backbone_atoms.index("CA")]
#     c = coords[:, backbone_atoms.index("C")]
#     cb = coords[:, len(backbone_atoms)]

#     mask = (cb.sum(-1) != 0.0) & (ca.sum(-1) != 0.0)
#     mask &= (c.sum(-1) != 0.0) & (n.sum(-1) != 0.0)

#     # Cahn Ingold Prelog Priority Rule, but using plane where Cb is behind
#     axis = normalize(ca - cb)

#     n_clock = (n - ca) - (axis * (n - ca)).sum(-1)[..., None] * axis
#     c_clock = (c - ca) - (axis * (c - ca)).sum(-1)[..., None] * axis

#     # https://stackoverflow.com/questions/14066933/
#     # direct-way-of-computing-clockwise-angle-between-2-vectors
#     determinant = (axis * np.cross(n_clock, c_clock)).sum(-1)
#     dot = (n_clock * c_clock).sum(-1)
#     angle = np.arctan2(determinant, dot)

#     mask &= np.isfinite(angle)
#     mean_chirality = (angle[mask] > 0.0).sum(-1) / (mask.sum(-1) + 1e-6)

#     return mean_chirality


# class MaybeMirror(MoleculeTransform):
#     def __init__(self, hand="left"):
#         self.hand = hand

#     def transform(self, datum):
#         mean_chirality = measure_chirality(datum.atom_coord)
#         datum_hand = "right" if (mean_chirality < 0.5) else "left"
#         if datum_hand != self.hand:
#             datum.atom_coord[..., 0] = (-1) * datum.atom_coord[..., 0]
#         return datum
