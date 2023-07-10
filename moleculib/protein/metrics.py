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

        return dict(
            num_clashes=num_clashes,
            avg_num_clashes=avg_num_clashes,
        )


class StandardBondDeviation(ProteinMetric):
    def __call__(self, datum: ProteinDatum):
        coords = rearrange(datum.atom_coord, "r a c -> (r a) c")
        i, j = rearrange(coords[datum.bonds_list], "... b c -> b ... c")
        norms = safe_norm((i - j)) * datum.bonds_mask
        error = jnp.square(norms - datum.bond_lens_list) * datum.bonds_mask
        error = error.sum((-1, -2)) / (datum.bonds_mask.sum((-1, -2)) + 1e-6)
        return dict(
            bond_deviation=error,
        )


if __name__ == "__main__":
    from moleculib.metrics import MetricsPipe
    from moleculib.protein.transform import ProteinCrop, DescribeChemistry
    from moleculib.protein.dataset import MonomerDataset

    data_path = "/mas/projects/molecularmachines/db/PDB"
    min_seq_len = 16
    max_seq_len = sequence_length = 512
    dataset = MonomerDataset(
        base_path=data_path,
        attrs="all",
        max_resolution=1.7,
        min_sequence_length=min_seq_len,
        max_sequence_length=max_seq_len,
        frac=1.0,
        transform=[
            ProteinCrop(crop_size=sequence_length),
            DescribeChemistry(),
        ],
    )

    datum = dataset[0]
    metrics_pipe = MetricsPipe([StandardBondDeviation(), CountClashes()])

    print(metrics_pipe(datum))