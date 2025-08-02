
import einops as ein
import jax.numpy as jnp
import jax
from typing import Tuple, Dict
from jaxtyping import PyTree
import optax

from ..protein.datum import ProteinDatum
from learnax.loss import LossFunction

from einops import rearrange



def safe_norm(vector: jax.Array, axis: int = -1) -> jax.Array:
    """safe_norm(x) = norm(x) if norm(x) != 0 else 1.0"""
    norms_sqr = jnp.sum(vector**2, axis=axis)
    norms = jnp.where(norms_sqr == 0.0, 1.0, norms_sqr) ** 0.5
    return norms


def safe_normalize(vector: jax.Array) -> jax.Array:
    return vector / safe_norm(vector)[..., None]


class ChemicalViolationLoss(LossFunction):
    def __init__(
        self,
        weight=1.0,
        key=None,
        measure=None,
    ):
        super().__init__(weight=weight)
        self.key = key
        self.measure = measure

    def _call(
        self, model_output: ProteinDatum, ground: ProteinDatum
    ) -> Tuple[ProteinDatum, jax.Array, Dict[str, float]]:
        if getattr(self, "key") is None:
            raise ValueError("Must set key before calling ChemicalViolationLoss")
        ground = ground[1]

        ground_coords = ground.atom_coord
        coords = model_output.atom_coord

        indices = getattr(ground, f"{self.key}_list")
        mask = getattr(ground, f"{self.key}_mask")

        target = self.measure(ground_coords, indices)
        prediction = self.measure(coords, indices)

        difference = target - prediction
        # if self.key == "dihedrals":
        #     alternative = (2 * jnp.pi - target) - prediction
        #     difference = jnp.where(
        #         jnp.abs(difference) < jnp.abs(alternative), difference, alternative
        #     )

        sqr_error = jnp.square(difference)
        sqr_error = sqr_error * mask.astype(sqr_error.dtype)
        mse = sqr_error.sum((-1, -2)) / (mask.sum((-1, -2)) + 1e-6)
        mse = mse.mean()
        mse = mse * (mask.sum() > 0).astype(mse.dtype)

        return model_output, mse, {f"{self.key}_loss": mse}


class BondLoss(ChemicalViolationLoss):
    def __init__(
        self,
        weight = 1.0,
    ):
        super().__init__(weight, "bonds", self.measure_bonds)

    @staticmethod
    def measure_bonds(coords, indices):
        coords = rearrange(coords, "r a c -> (r a) c")
        i, j = rearrange(coords[indices], "... b c -> b ... c")
        norms = safe_norm((i - j))
        return norms


class AngleLoss(ChemicalViolationLoss):
    def __init__(
        self,
        weight = 1.0,
    ):
        super().__init__(weight, "angles", self.measure_angles)

    @staticmethod
    def measure_angles(coords, indices):
        coords = rearrange(coords, "r a c -> (r a) c")
        i, j, k = rearrange(indices, "... b -> b ...")

        v1, v2 = coords[i] - coords[j], coords[k] - coords[j]
        v1, v2 = safe_normalize(v1), safe_normalize(v2)
        x, y = safe_norm(v1 + v2), safe_norm(v1 - v2)

        x = jnp.where(x == 0.0, 1e-6, x)
        # NOTE(Allan): this might throw errors still
        # jax._src.source_info_util.JaxStackTraceBeforeTransformation:
        # FloatingPointError: invalid value (nan) encountered in jit(div)
        return 2 * jnp.arctan2(y, x)


class DihedralLoss(ChemicalViolationLoss):
    def __init__(
        self,
        weight,
        start_step,
    ):
        super().__init__(weight, start_step, "dihedrals", self.measure_dihedrals)

    @staticmethod
    def measure_dihedrals(coords, indices):
        # based on Hypnopump's Gist
        coords = rearrange(coords, "r a c -> (r a) c")
        p, q, v, u = rearrange(indices, "... b -> b ...")
        a1 = coords[q] - coords[p]
        a2 = coords[v] - coords[q]
        a3 = coords[u] - coords[v]

        v1 = jnp.cross(a1, a2)
        v1 = safe_normalize(v1)
        v2 = jnp.cross(a2, a3)
        v2 = safe_normalize(v2)

        rad = 2 * jnp.arctan2(safe_norm(v1 - v2), safe_norm(v1 + v2))

        return rad


import e3nn_jax as e3nn

class ClashLoss(LossFunction):

    def __call__(self, model_output: ProteinDatum, ground: ProteinDatum):
        ground = ground[1]

        coords = model_output.atom_coord
        all_atom_coords = rearrange(coords, "r a c -> (r a) c")
        all_atom_radii = rearrange(ground.atom_radius, "r a -> (r a)")
        all_atom_mask = rearrange(ground.atom_mask, "r a -> (r a)")

        vector_map = lambda x: rearrange(x, "i c -> i () c") - rearrange(
            x, "j c -> () j c"
        )

        def mask_bonds(mask, bonds, bonds_mask):
            bonds = jnp.where(bonds_mask[..., None], bonds, coords.shape[-3] * 14)
            i, j = rearrange(bonds, "... b -> b (...)")
            mask = mask.at[i, j].set(False, mode="drop")
            return mask

        distance_maps = safe_norm(vector_map(all_atom_coords))
        cross_radii = rearrange(all_atom_radii, "i -> i ()") + rearrange(
            all_atom_radii, "j -> () j"
        )

        cross_mask = rearrange(all_atom_mask, "i -> i ()") & rearrange(
            all_atom_mask, "j -> () j"
        )
        cross_mask = mask_bonds(cross_mask, ground.bonds_list, ground.bonds_mask)

        cross_radii = jnp.where(cross_radii == 0.0, 1.0, cross_radii)
        clashes = e3nn.soft_envelope(
            distance_maps,
            x_max=cross_radii,
            arg_multiplicator=10.0,
            value_at_origin=1.0,
        )

        mse = (clashes * cross_mask).sum((-1, -2)) / (cross_mask.sum((-1, -2)) + 1e-6)
        mse = mse.mean()
        mse = mse * (cross_mask.sum() > 0).astype(mse.dtype)

        return model_output, mse, {"clash_loss": mse}
