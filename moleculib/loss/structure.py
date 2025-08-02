import einops as ein
import jax.numpy as jnp
import jax
from typing import Tuple, Dict
from jaxtyping import PyTree
import optax

from ..protein.datum import ProteinDatum

def safe_norm(vector: jax.Array, axis: int = -1) -> jax.Array:
    """safe_norm(x) = norm(x) if norm(x) != 0 else 1.0"""
    norms_sqr = jnp.sum(vector**2, axis=axis)
    norms = jnp.where(norms_sqr == 0.0, 1.0, norms_sqr) ** 0.5
    return norms


class VectorMapLoss:

    def __init__(
        self,
        max_radius: float = 25.0,
        max_error: float = 10.0,
        measure_fn: jax.Array = optax.huber_loss,
        norm_only=False,
    ):
        self.norm_only = norm_only
        self.max_radius = max_radius
        self.max_error = max_error
        self.measure_fn = measure_fn

    def __call__(
        self, prediction: ProteinDatum, ground: ProteinDatum
    ) -> Tuple[ProteinDatum, jax.Array, Dict[str, float]]:

        if type(ground) == list or type(ground) == tuple and len(ground) == 2:
            ground = ground[1]

        all_atom_coords = ein.rearrange(prediction.atom_coord, "... a c -> (... a) c")
        all_atom_coords_ground = ein.rearrange(ground.atom_coord, "... a c -> (... a) c")
        all_atom_mask = ein.rearrange(ground.atom_mask, "... a -> (... a)")

        vector_map = lambda x: ein.rearrange(x, "i c -> i () c") - ein.rearrange(
            x, "j c -> () j c"
        )

        cross_mask = ein.rearrange(all_atom_mask, "i -> i ()") & ein.rearrange(
            all_atom_mask, "j -> () j"
        )

        vector_maps = vector_map(all_atom_coords)
        vector_maps_ground = vector_map(all_atom_coords_ground)
        cross_mask = cross_mask & (safe_norm(vector_maps_ground) < self.max_radius)

        if self.norm_only:
            vector_maps = safe_norm(vector_maps)[..., None]
            vector_maps_ground = safe_norm(vector_maps_ground)[..., None]

        error = self.measure_fn(vector_maps, vector_maps_ground).mean(-1)
        if self.max_error > 0.0:
            error = jnp.clip(error, 0.0, self.max_error)

        error = (error * cross_mask.astype(error.dtype)).sum((-1, -2)) / (
            cross_mask.sum((-1, -2)) + 1e-6
        )
        error = error.mean()
        error = error * (cross_mask.sum() > 0).astype(error.dtype)

        metrics = dict(
            vector_map_loss=error,
        )

        return prediction, error, metrics
