from .datum import ProteinDatum
from .alphabet import (
    all_atoms,
    all_atoms_elements,
    all_atoms_radii,
    backbone_atoms,
    bonds_arr,
    bonds_mask,
    angles_arr,
    angles_mask,
    dihedrals_arr,
    dihedrals_mask,
    flippable_arr,
    flippable_mask,
)
import numpy as np
from biotite.sequence import ProteinSequence
from einops import rearrange
from .utils import pad_array
from functools import partial


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
    def __init__(self, crop_size, pad=True):
        self.crop_size = crop_size
        if pad:
            self.padding = partial(pad_array, total_size=crop_size)

    def transform(self, datum, cut=None):
        seq_len = datum.residue_index.shape[0]
        if seq_len == self.crop_size:
            return datum
        elif seq_len < self.crop_size and hasattr(self, "padding"):
            new_datum = ProteinDatum(
                idcode=datum.idcode,
                resolution=datum.resolution,
                sequence=datum.sequence,
                residue_index=self.padding(datum.residue_index),
                residue_token=self.padding(datum.residue_token),
                residue_mask=self.padding(datum.residue_mask),
                chain_token=self.padding(datum.chain_token),
                atom_token=self.padding(datum.atom_token),
                atom_coord=self.padding(datum.atom_coord),
                atom_mask=self.padding(datum.atom_mask),
            )
        else:
            if cut is None:
                cut = np.random.randint(low=0, high=(seq_len - self.crop_size))
            new_datum = ProteinDatum(
                idcode=datum.idcode,
                resolution=datum.resolution,
                sequence=datum.sequence[cut : cut + self.crop_size],
                residue_index=datum.residue_index[cut : cut + self.crop_size],
                residue_token=datum.residue_token[cut : cut + self.crop_size],
                residue_mask=datum.residue_mask[cut : cut + self.crop_size],
                chain_token=datum.chain_token[cut : cut + self.crop_size],
                atom_token=datum.atom_token[cut : cut + self.crop_size],
                atom_coord=datum.atom_coord[cut : cut + self.crop_size],
                atom_mask=datum.atom_mask[cut : cut + self.crop_size],
            )

        return new_datum


class ListBonds(ProteinTransform):
    def transform(self, datum):
        # solve for intra-residue bonds
        num_atoms = datum.atom_coord.shape[-2]
        count = num_atoms * np.expand_dims(
            np.arange(0, len(datum.residue_token)), axis=(-1, -2)
        )
        bonds_per_residue = bonds_arr[datum.residue_token]

        bond_list = (bonds_per_residue + count).astype(np.int32)
        bond_mask = bonds_mask[datum.residue_token].squeeze(-1)
        bond_list[~bond_mask] = 0

        # add peptide bonds
        n_page = backbone_atoms.index("N")
        ns = num_atoms * np.arange(1, len(datum.atom_coord)) + n_page
        c_page = backbone_atoms.index("C")
        cs = num_atoms * np.arange(0, len(datum.atom_coord) - 1) + c_page

        peptide_bonds = np.stack((ns, cs)).T
        peptide_mask = np.ones(peptide_bonds.shape[:-1], dtype=np.bool_)

        peptide_bonds = np.pad(peptide_bonds, ((0, 1), (0, 0)), constant_values=0)
        peptide_mask = np.pad(peptide_mask, ((0, 1)), constant_values=False)

        bond_list = np.concatenate((bond_list, peptide_bonds[:, None, :]), axis=1)
        bond_mask = np.concatenate((bond_mask, peptide_mask[:, None]), axis=1)

        left, right = rearrange(bond_list, "s b i -> i (s b)")
        atom_mask = rearrange(datum.atom_mask, "r a -> (r a)")
        bond_mask &= rearrange(
            atom_mask[left] & atom_mask[right], "(s b) -> s b", b=bond_list.shape[1]
        )

        datum.bonds_list = bond_list
        datum.bonds_mask = bond_mask

        return datum


class ListAngles(ProteinTransform):
    def transform(self, datum):
        # solve for intra-residue angles
        num_atoms = datum.atom_coord.shape[-2]
        count = num_atoms * np.expand_dims(
            np.arange(0, len(datum.residue_token)), axis=(-1, -2)
        )
        angles_per_residue = angles_arr[datum.residue_token]

        angle_mask = angles_mask[datum.residue_token].squeeze(-1)
        angle_list = (angles_per_residue + count).astype(np.int32)
        angle_list[~angle_mask] = 0

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

        peptide_angles = np.stack((first_angle, second_angle), axis=1)
        peptide_mask = np.ones(peptide_angles.shape[:-1], dtype=np.bool_)

        peptide_angles = np.pad(
            peptide_angles, ((0, 1), (0, 0), (0, 0)), constant_values=0
        )
        peptide_mask = np.pad(peptide_mask, ((0, 1), (0, 0)), constant_values=False)

        angle_list = np.concatenate((angle_list, peptide_angles), axis=1)
        angle_mask = np.concatenate((angle_mask, peptide_mask), axis=1)

        # kill bonds for which atom_mask flags lack record
        i, j, k = rearrange(angle_list, "s a i -> i (s a)")
        atom_mask = rearrange(datum.atom_mask, "r a -> (r a)")
        angle_mask &= rearrange(
            atom_mask[i] & atom_mask[j] & atom_mask[k],
            "(s b) -> s b",
            b=angle_list.shape[1],
        )

        datum.angles_list = angle_list
        datum.angles_mask = angle_mask

        return datum


class ListDihedrals(ProteinTransform):
    def transform(self, datum):
        # solve for intra-residue angles
        num_atoms = datum.atom_coord.shape[-2]
        count = num_atoms * np.expand_dims(
            np.arange(0, len(datum.residue_token)), axis=(-1, -2)
        )

        dihedrals_per_residue = dihedrals_arr[datum.residue_token]

        dihedral_mask = dihedrals_mask[datum.residue_token].squeeze(-1)
        dihedral_list = (dihedrals_per_residue + count).astype(np.int32)
        dihedral_list[~dihedral_mask] = 0

        def prev_(atom):
            page = num_atoms * np.arange(0, len(datum.atom_coord) - 1)
            return page + all_atoms.index(atom)

        def next_(atom):
            page = num_atoms * np.arange(1, len(datum.atom_coord))
            return page + all_atoms.index(atom)

        psi = np.stack((prev_("N"), prev_("CA"), prev_("C"), next_("N"))).T
        omega = np.stack((prev_("CA"), prev_("C"), next_("N"), next_("CA"))).T
        phi = np.stack((prev_("C"), next_("N"), next_("CA"), next_("C"))).T

        peptide_dihedrals = np.stack((psi, phi, omega), axis=1)
        peptide_mask = np.ones(peptide_dihedrals.shape[:-1], dtype=np.bool_)

        peptide_dihedrals = np.pad(
            peptide_dihedrals, ((0, 1), (0, 0), (0, 0)), constant_values=0
        )
        peptide_mask = np.pad(peptide_mask, ((0, 1), (0, 0)), constant_values=False)

        dihedral_list = np.concatenate((dihedral_list, peptide_dihedrals), axis=1)
        dihedral_mask = np.concatenate((dihedral_mask, peptide_mask), axis=1)

        # kill dihedrals for which atom_mask flags lack record
        p, q, u, v = rearrange(dihedral_list, "s d i -> i (s d)")
        atom_mask = rearrange(datum.atom_mask, "r a -> (r a)")
        dihedral_mask &= rearrange(
            atom_mask[p] & atom_mask[q] & atom_mask[u] & atom_mask[v],
            "(s d) -> s d",
            d=dihedral_list.shape[1],
        )

        datum.dihedrals_list = dihedral_list
        datum.dihedrals_mask = dihedral_mask

        return datum


class ListMirrorFlips(ProteinTransform):
    def transform(self, datum):
        # solve for intra-residue angles
        num_atoms = datum.atom_coord.shape[-2]
        flips_per_residue = flippable_arr[datum.residue_token]

        flips_mask_per_residue = flippable_mask[datum.residue_token].squeeze(-1)
        flips_list = (flips_per_residue).astype(np.int32)

        datum.flips_list = flips_list
        datum.flips_mask = flips_mask_per_residue

        return datum


class DescribeChemistry(ProteinTransform):
    """
    Augments ProteinDatum with bonds_list and angles_list a list of atomic indexes connected by bonds and angles
    that should have invariant behavior according to chemistry
    Note that indexing is performed at atom level, that is, as residue_dim atom_dim -> (residue_dim atom_dim)
    """

    def __init__(self):
        self.bond_transform = ListBonds()
        self.angle_transform = ListAngles()
        self.dihedral_transform = ListDihedrals()
        self.flip_transform = ListMirrorFlips()

    def transform(self, datum):
        datum.atom_token = datum.atom_token.astype(np.int32)
        datum.atom_element = all_atoms_elements[datum.atom_token]
        datum.atom_radius = all_atoms_radii[datum.atom_token]
        datum = self.bond_transform.transform(datum)
        datum = self.angle_transform.transform(datum)
        datum = self.dihedral_transform.transform(datum)
        datum = self.flip_transform.transform(datum)
        return datum


class TokenizeSequenceBoundaries(ProteinTransform):
    """
    Augments ProteinDatum with boundary_token and boundary_mask
    """

    def transform(self, datum):
        boundary_token = np.zeros(len(datum.residue_token), dtype=np.int32)
        boundary_mask = np.zeros(len(datum.residue_token), dtype=np.bool_)
        boundary_token[0] = 1
        boundary_token[-1] = 2

        boundary_mask[0] = True & datum.atom_mask[0, 1]
        boundary_mask[-1] = True & datum.atom_mask[-1, 1]

        datum.boundary_token = boundary_token
        datum.boundary_mask = boundary_mask

        return datum


def normalize(vector):
    norms_sqr = np.sum(vector**2, axis=-1, keepdims=True)
    norms = np.where(norms_sqr == 0.0, 1.0, norms_sqr) ** 0.5
    return vector / norms


def measure_chirality(coords):
    n = coords[:, backbone_atoms.index("N")]
    ca = coords[:, backbone_atoms.index("CA")]
    c = coords[:, backbone_atoms.index("C")]
    cb = coords[:, len(backbone_atoms)]

    mask = (cb.sum(-1) != 0.0) & (ca.sum(-1) != 0.0)
    mask &= (c.sum(-1) != 0.0) & (n.sum(-1) != 0.0)

    # Cahn Ingold Prelog Priority Rule, but using plane where Cb is behind
    axis = normalize(ca - cb)

    n_clock = (n - ca) - (axis * (n - ca)).sum(-1)[..., None] * axis
    c_clock = (c - ca) - (axis * (c - ca)).sum(-1)[..., None] * axis

    # https://stackoverflow.com/questions/14066933/
    # direct-way-of-computing-clockwise-angle-between-2-vectors
    determinant = (axis * np.cross(n_clock, c_clock)).sum(-1)
    dot = (n_clock * c_clock).sum(-1)
    angle = np.arctan2(determinant, dot)

    mask &= np.isfinite(angle)
    mean_chirality = (angle[mask] > 0.0).sum(-1) / (mask.sum(-1) + 1e-6)

    return mean_chirality


class MaybeMirror(ProteinTransform):
    def __init__(self, hand="left"):
        self.hand = hand

    def transform(self, datum):
        try:
            mean_chirality = measure_chirality(datum.atom_coord)
        except:
            breakpoint()
        datum_hand = "right" if (mean_chirality < 0.5) else "left"
        if datum_hand != self.hand:
            datum.atom_coord[..., 0] = (-1) * datum.atom_coord[..., 0]
        return datum


class RemoveUnks(ProteinTransform):
    def __init__(self):
        super().__init__()

    def transform(self, datum):
        # identify non-unk indices
        idx = [i for i in range(len(datum.sequence)) if datum.sequence[i] != 'X']

        # construct new sequence without unks
        old_seq = str(datum.sequence)
        new_seq = [old_seq[i] for i in range(len(old_seq)) if i in idx]
        new_seq = ProteinSequence(new_seq)

        # construct all data attrs accordingly
        ri = np.arange(0, len(new_seq))
        rt = datum.residue_token[idx]
        rm = datum.residue_mask[idx]
        ct = datum.chain_token[idx]
        at = datum.atom_token[idx]
        ac = datum.atom_coord[idx]
        am = datum.atom_mask[idx]

        # build new datum
        new_datum = ProteinDatum(
            idcode=datum.idcode,
            resolution=datum.resolution,
            sequence=new_seq,
            residue_index=ri,
            residue_token=rt,
            residue_mask=rm,
            chain_token=ct,
            atom_token=at,
            atom_coord=ac,
            atom_mask=am,
        )

        return new_datum
