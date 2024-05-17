from einops import rearrange
import numpy as np

from .measures import STANDARD_CHEMICAL_MEASURES

from moleculib.protein.datum import ProteinDatum
import jax.numpy as jnp
from typing import List


def safe_norm(vector: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """safe_norm(x) = norm(x) if norm(x) != 0 else 1.0"""
    norms_sqr = jnp.sum(vector**2, axis=axis)
    norms = jnp.where(norms_sqr == 0.0, 1.0, norms_sqr) ** 0.5
    return norms


class ProteinMetric:
    def __call__(self, datum: ProteinDatum):
        raise NotImplementedError("ProteinMetric is abstract")


class AlignedRootMeanSquareDeviation(ProteinMetric):

    def __call__(self, datum, other_datum):
        datum = datum.align_to(other_datum)
        other_datum = other_datum.align_to(datum)
        other_ca_coord = other_datum.atom_coord[..., 1, :]
        ca_coord = datum.atom_coord[..., 1, :]
        mask = other_datum.atom_mask[..., 1] & datum.atom_mask[..., 1]
        diff = jnp.square(ca_coord - other_ca_coord).sum(-1)
        diff = diff[mask]
        diff = diff.sum()
        diff = diff / (mask.sum() + 1e-6)
        return {"rmsd": diff ** 0.5}


class CountClashes(ProteinMetric):
    def __init__(self, radius_multiplier=0.8, smooth: bool = False):
        self.radius_multiplier = radius_multiplier
        self.smooth = smooth

    def __call__(self, datum: ProteinDatum):
        seq_len = datum.atom_coord.shape[-3]
        all_atom_coords = rearrange(datum.atom_coord, "r a c -> (r a) c")
        all_atom_radii = rearrange(datum.atom_radius, "r a -> (r a)")
        all_atom_mask = rearrange(datum.atom_mask, "r a -> (r a)")

        vector_map = rearrange(all_atom_coords, "i c ->  i () c") - rearrange(
            all_atom_coords, "j c -> () j c"
        )

        def mask_bonds(mask, bonds, bonds_mask):
            bonds = jnp.where(bonds_mask[..., None], bonds, seq_len * 14)
            i, j = rearrange(bonds, "... b -> b (...)")
            mask = mask.at[i, j].set(False, mode="drop")
            mask = mask.at[j, i].set(False, mode="drop")
            return mask

        distance_maps = safe_norm(vector_map)
        cross_radii = rearrange(all_atom_radii, " i -> i ()") + rearrange(
            all_atom_radii, "j -> () j"
        )

        cross_mask = rearrange(all_atom_mask, "i -> i ()") & rearrange(
            all_atom_mask, "... j -> ... () j"
        )
        cross_mask = mask_bonds(
            jnp.array(cross_mask), datum.bonds_list, datum.bonds_mask
        )
        cross_radii = jnp.where(cross_radii == 0.0, 1.0, cross_radii)

        is_clash = distance_maps < cross_radii * self.radius_multiplier
        num_clashes = (is_clash * cross_mask).sum((-1, -2)) / 2
        avg_num_clashes = num_clashes / (cross_mask.sum((-1, -2)) + 1e-6)

        return dict(
            num_clashes=num_clashes,
            avg_num_clashes=avg_num_clashes,
        )


def norm(vector: np.ndarray) -> np.ndarray:
    norms_sqr = jnp.sum(vector**2, axis=-1)
    norms = norms_sqr**0.5
    return norms


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / safe_norm(vector)[..., None]


def measure_bonds(coord, idx):
    v, u = idx.T
    bonds_len = jnp.sqrt(jnp.square(coord[v] - coord[u]).sum(-1))
    return bonds_len


def measure_angles(coords, idx):
    i, j, k = rearrange(idx, "... a -> a ...")
    v1, v2 = coords[i] - coords[j], coords[k] - coords[j]
    v1, v2 = normalize(v1), normalize(v2)
    x, y = norm(v1 + v2), norm(v1 - v2)
    return 2 * jnp.arctan2(y, x)


def measure_dihedrals(coords, indices):
    p, q, v, u = rearrange(indices, "... b -> b ...")
    u1, u2, u3, u4 = coords[p], coords[q], coords[v], coords[u]

    a1 = u2 - u1
    a2 = u3 - u2
    a3 = u4 - u3

    v1 = jnp.cross(a1, a2)
    v1 = normalize(v1)
    v2 = jnp.cross(a2, a3)
    v2 = normalize(v2)

    porm = jnp.sign((v1 * a3).sum(-1))
    rad = jnp.arccos((v1 * v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1)) ** 0.5)
    rad = jnp.where(porm == 0, rad * porm, rad)

    mask = (
        (u1.sum(-1) != 0.0)
        & (u2.sum(-1) != 0.0)
        & (u3.sum(-1) != 0.0)
        & (u4.sum(-1) != 0.0)
    )

    return rad * mask



class ChemicalDeviationMetric(ProteinMetric):
    def __init__(self, key, measure, var_clip=0.0, num_interactive_atoms=1):
        self.key = key
        self.measure = measure
        self.var_clip = var_clip
        self.num_interactive_atoms = num_interactive_atoms

    def __call__(self, datum: ProteinDatum):
        coords = rearrange(datum.atom_coord, "r a c -> (r a) c")
        idx = rearrange(getattr(datum, f"{self.key}_list"), "r a c -> (r a) c")

        res_token = datum.residue_token.astype(jnp.int32)
        standard_values = jnp.array(STANDARD_CHEMICAL_MEASURES[self.key][0])[res_token]

        mask = getattr(datum, f"{self.key}_mask").astype(jnp.float32)
        if self.var_clip > 0.0:
            standard_vars = jnp.array(STANDARD_CHEMICAL_MEASURES[self.key][1])[
                res_token
            ]
            mask = mask * (standard_vars < self.var_clip).astype(jnp.float32)

        values = self.measure(coords, idx)

        mask = jnp.where(jnp.isnan(mask), 0.0, mask)
        values = jnp.where(jnp.isnan(values), 0.0, values)

        values = rearrange(values, "(r a) -> r a", r=datum.residue_token.shape[0])

        error = jnp.square(values - standard_values) * mask

        error_internal = error[..., :-self.num_interactive_atoms]
        mask_internal = mask[..., :-self.num_interactive_atoms]

        error_external = error[..., -self.num_interactive_atoms:]
        mask_external = mask[..., -self.num_interactive_atoms:]

        error_internal = error_internal.sum((-1, -2)) / (mask_internal.sum((-1, -2)) + 1e-6)
        error_external = error_external.sum((-1, -2)) / (mask_external.sum((-1, -2)) + 1e-6)

        out = dict()
        out[f"peptide_{self.key}_deviation"] = error_external
        out[f"internal_{self.key}_deviation"] = error_internal

        return out


class StandardBondDeviation(ChemicalDeviationMetric):
    def __init__(self):
        super().__init__("bonds", measure_bonds, num_interactive_atoms=1)


class StandardAngleDeviation(ChemicalDeviationMetric):
    def __init__(self):
        super().__init__("angles", measure_angles, num_interactive_atoms=2)



class StandardDihedralDeviation(ChemicalDeviationMetric):
    def __init__(self, var_clip=0.1):
        super().__init__("dihedrals", measure_dihedrals, var_clip=var_clip, num_interactive_atoms=3)
        

from moleculib.protein.transform import (
    DescribeChemistry
)


class StandardChemicalDeviation(ProteinMetric):
    def __init__(self):
        self.describe_chemistry = DescribeChemistry()
        self.count_clashes = CountClashes()
        self.bond_deviation = StandardBondDeviation()
        self.angle_deviation = StandardAngleDeviation()
        self.dihedral_deviation = StandardDihedralDeviation()

    def __call__(self, datum: ProteinDatum):
        if not hasattr(datum, "bonds_list"):
            datum = self.describe_chemistry.transform(datum)

        bond_deviation = self.bond_deviation(datum)
        angle_deviation = self.angle_deviation(datum)
        dihedral_deviation = self.dihedral_deviation(datum)
        clashes_count = self.count_clashes(datum)

        return {
            **bond_deviation,
            **angle_deviation,
            **dihedral_deviation,
            **clashes_count,
        }
