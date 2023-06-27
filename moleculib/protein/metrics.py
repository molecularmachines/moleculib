from einops import rearrange
import numpy as np

from moleculib.protein.datum import ProteinDatum
import jax.numpy as jnp


def safe_norm(vector: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """safe_norm(x) = norm(x) if norm(x) != 0 else 1.0"""
    norms_sqr = jnp.sum(vector**2, axis=axis)
    norms = jnp.where(norms_sqr == 0.0, 1.0, norms_sqr) ** 0.5
    return norms


class ProteinMetric:
    def __call__(self, datum: ProteinDatum):
        raise NotImplementedError("ProteinMetric is abstract")


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
        num_clashes = (is_clash * cross_mask).sum((-1, -2))
        avg_num_clashes = num_clashes / (cross_mask.sum((-1, -2)) + 1e-6)

        # if self.smooth:
        #     clashes = e3nn.soft_envelope(
        #         distance_maps,
        #         x_max=cross_radii,
        #         arg_multiplicator=10.0,
        #         value_at_origin=1.0,
        #     )

        return dict(
            num_clashes=num_clashes,
            avg_num_clashes=avg_num_clashes,
        )


# class BondDeviation(ProteinMetric):
#     def __call__(self, datum: ProteinDatum):
#         coords = rearrange(coords, "r a c -> (r a) c")
#         i, j = rearrange(coords[indices], "... b c -> b ... c")
#         norms = safe_norm((i - j))
#         return norms
#         indices, mask = ground[f"{self.key}_list"], ground[f"{self.key}_mask"]

#         target = jax.vmap(self.measure)(ground_coords, indices)
#         prediction = jax.vmap(self.measure)(coords, indices)

#         difference = target - prediction
#         if self.key == "dihedrals":
#             alternative = (2 * jnp.pi - target) - prediction
#             difference = jnp.where(
#                 jnp.abs(difference) < jnp.abs(alternative), difference, alternative
#             )

#         sqr_error = jnp.square(difference)
#         sqr_error = sqr_error * mask.astype(sqr_error.dtype)
#         mse = sqr_error.sum((-1, -2)) / (mask.sum((-1, -2)) + 1e-6)
#         mse = mse.mean()

#         return model_output, mse, {f"{self.key}_loss": mse}


if __name__ == "__main__":
    from moleculib.metrics import MetricsPipe
    from moleculib.protein.transform import DescribeChemistry

    metrics_pipe = MetricsPipe([CountClashes()])
    datum = ProteinDatum.fetch_pdb_id("4AKE")
    datum = DescribeChemistry().transform(datum)

    print(metrics_pipe(datum))
