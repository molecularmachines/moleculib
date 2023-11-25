import math

import biotite
from .datum import ProteinDatum
from .alphabet import (
    all_atoms,
    all_residues,
    all_atoms_elements,
    all_atoms_radii,
    backbone_atoms,
    bonds_arr,
    bond_lens_arr,
    bonds_mask,
    angles_arr,
    angles_mask,
    dihedrals_arr,
    dihedrals_mask,
    flippable_arr,
    flippable_mask,
)
import numpy as np
from einops import rearrange
from .utils import pad_array

import jax.numpy as jnp


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

    def transform(self, datum, cut=None):
        seq_len = datum.residue_token.shape[0]
        if seq_len <= self.crop_size:
            return datum
        if cut is None:
            cut = np.random.randint(low=0, high=(seq_len - self.crop_size))

        new_datum_ = dict()
        for attr, obj in vars(datum).items():
            if type(obj) in [np.ndarray, list, tuple, str] and len(obj) == seq_len:
                new_datum_[attr] = obj[cut : cut + self.crop_size]
            else:
                new_datum_[attr] = obj

        new_datum = ProteinDatum(**new_datum_)
        return new_datum


class ProteinRescale(ProteinTransform):
    def __init__(self, factor):
        self.factor = factor

    def transform(self, datum):
        coord = datum.atom_coord
        mask = datum.atom_mask
        all_coords = rearrange(coord, "r a c -> (r a) c")
        all_masks = rearrange(mask, "r a -> (r a)")

        center = (all_coords * all_masks[..., None]).sum(0) / all_masks[..., None].sum(
            axis=0
        )
        datum.atom_coord = coord - center[..., None, :] * mask[..., None]

        datum.atom_coord = datum.atom_coord / self.factor
        return datum


class BackboneOnly(ProteinTransform):
    def __init__(self, filter: bool = True, keep_seq: bool = False):
        self.filter = filter
        self.keep_seq = keep_seq
        
    def transform(self, datum):
        if self.filter:
            datum.atom_coord[..., 4:, :] = 0.0
            datum.atom_mask[..., 4:] = False
            if not self.keep_seq:
                datum.residue_token[datum.residue_token > 2] = 10  # GLY
        return datum


class CaOnly(ProteinTransform):
    def __init__(self, filter: bool = True, keep_seq: bool = False):
        self.filter = filter
        self.keep_seq = keep_seq
        
    def transform(self, datum):
        if self.filter:
            datum.atom_mask[..., 0] = False
            datum.atom_coord[..., 0, :] = 0.0

            datum.atom_coord[..., 2:, :] = 0.0
            datum.atom_mask[..., 2:] = False

            if not self.keep_seq:
                datum.residue_token[datum.residue_token > 2] = 10  # GLY

        return datum
    
from einops import repeat

class ToJraph(ProteinTransform):
    def __init__(self):
        import jraph
        self.jraph = jraph

    def transform(self, datum: ProteinDatum):
        num_nodes = len(datum.residue_token[datum.pad_mask.astype(jnp.bool_)])
        # node_features = datum.residue_token[datum.pad_mask.astype(jnp.bool_)]
        node_indices = jnp.arange(len(datum))
        n_node = jnp.array([num_nodes])
        node_pos = datum.atom_coord[..., 1, :][datum.pad_mask.astype(jnp.bool_)]
        senders = repeat(node_indices, "i -> (i j)", j=num_nodes)
        receivers = repeat(node_indices, "i -> (j i)", j=num_nodes)
        graph = self.jraph.GraphsTuple(nodes=node_pos, edges=None, senders=senders, receivers=receivers, n_node=n_node, n_edge=jnp.array([num_nodes**2]), globals=None)
        if num_nodes < 63:
            graph = self.jraph.pad_with_graphs(graph, n_node=63, n_edge=63**2)
        return graph

class ProteinPad(ProteinTransform):
    def __init__(self, pad_size: int, random_position: bool = False):
        self.pad_size = pad_size
        self.random_position = random_position

    def transform(self, datum: ProteinDatum) -> ProteinDatum:
        seq_len = datum.residue_token.shape[0]
        if seq_len >= self.pad_size:
            datum.pad_mask = np.ones_like(datum.residue_token)
            return datum

        size_diff = self.pad_size - seq_len
        shift = np.random.randint(0, size_diff)

        new_datum_ = dict()
        for attr, obj in vars(datum).items():
            if type(obj) == np.ndarray and attr != "label" and len(obj) == seq_len:
                obj = pad_array(obj, self.pad_size)
                if self.random_position:
                    obj = np.roll(obj, shift, axis=0)
                    if attr in ["bonds_list", "angles_list", "dihedrals_list"]:
                        obj += shift * 14
                new_datum_[attr] = obj
            else:
                new_datum_[attr] = obj

        pad_mask = pad_array(np.ones_like(datum.residue_token), self.pad_size)
        if self.random_position:
            pad_mask = np.roll(pad_mask, shift, axis=0)
        new_datum_["pad_mask"] = pad_mask

        new_datum = ProteinDatum(**new_datum_)
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
        # NOTE(Allan): need a better interface for specifying peptide bonds

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
        if len(boundary_token) != 0:
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


class CastToBFloat(ProteinTransform):
    def transform(self, datum):
        new_datum_ = dict()
        for attr, obj in vars(datum).items():
            if type(obj) == np.ndarray and (obj.dtype in [np.float32, np.float64]):
                obj = obj.astype(jnp.bfloat16)
            new_datum_[attr] = obj
        return ProteinDatum(**new_datum_)


SSE_TOKENS = ["", "c", "a", "b"]


class AnnotateSecondaryStructure(ProteinTransform):
    def transform(self, datum: ProteinDatum):
        coords = datum.atom_coord[..., 1, :]
        array = biotite.structure.array(
            [biotite.structure.Atom(coord, chain_id="A") for coord in coords]
        )
        array.set_annotation("atom_name", ["CA" for _ in datum.residue_token])
        array.set_annotation(
            "res_name", [all_residues[token] for token in datum.residue_token]
        )
        array.set_annotation("res_id", np.arange(0, len(coords)))
        annotations = biotite.structure.annotate_sse(array, chain_id="A")

        present, count = np.unique(annotations, return_counts=True)
        tokenized_count = np.zeros(4, dtype=np.int32)
        for idx, char in enumerate(present):
            tokenized_count[SSE_TOKENS.index(char)] = count[idx]

        annotations = [SSE_TOKENS.index(token) for token in annotations]
        datum.sse_token = np.array(annotations, dtype=np.int32)
        datum.sse_count = tokenized_count

        return datum


class MaskResidues(ProteinTransform):
    def __init__(self, mask_ratio: float = 0.0, contiguous: float = 0.0):
        self.mask_ratio = mask_ratio
        self.contiguous = contiguous
        
    def transform(self, datum: ProteinDatum, mask=None):
        if mask is not None:
            pass
        elif self.contiguous > 0.0:
            num_units = int(np.round(np.random.exponential(self.contiguous)))
            frac = np.round(len(datum.residue_token)*self.mask_ratio)
            if num_units >= frac:
                mask = np.random.rand(len(datum.residue_token)) < self.mask_ratio   
            else:
                if num_units > 1:
                    sizes = np.sort(np.random.choice(np.arange(1,frac), size=num_units-1, replace=False))
                    sizes = np.concatenate((np.array([0]), sizes, np.array([frac])))
                    sizes = np.diff(sizes)
                else:
                    sizes = np.array([frac])
                pos = np.sort(np.random.choice(np.arange(len(datum.residue_token)), size=num_units, replace=False))
                mask = np.zeros(len(datum.residue_token), dtype=np.bool_)
                for p, s in zip(pos, sizes):
                    mask[int(p):min(int(p+s),len(datum.residue_token))] = True
        else:
            num_mask = math.ceil(self.mask_ratio * datum.residue_token.shape[0])
            mask = np.zeros_like(datum.residue_token, dtype=np.bool_)
            choice = np.random.choice(len(datum.residue_token), num_mask, replace=False)
            mask[choice] = True

        mask = mask * datum.residue_mask
        datum.residue_token_masked = np.where(
            mask, all_residues.index("MASK"), datum.residue_token
        )
        datum.atom_coord_masked = datum.atom_coord * (1 - mask[:, None, None])
        datum.mask_mask = mask
        return datum
