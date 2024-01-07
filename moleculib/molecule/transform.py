from .datum import MoleculeDatum
import numpy as np
from .utils import pad_array
from functools import partial
import jax.numpy as jnp
from scipy.sparse.csgraph import laplacian


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
    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def transform(self, datum: MoleculeDatum, cut=None) -> MoleculeDatum:
        seq_len = len(datum.atom_token)
        if seq_len <= self.crop_size:
            return datum
        if cut is None:
            cut = np.random.randint(low=0, high=(seq_len - self.crop_size))

        new_datum_ = dict()
        for attr, obj in vars(datum).items():
            if (
                type(obj) in [np.ndarray, list, tuple, str]
                and len(obj) == seq_len
                and attr != "bonds"
            ):
                new_datum_[attr] = obj[cut : cut + self.crop_size]
            else:
                new_datum_[attr] = obj

        bonds = new_datum_["bonds"]
        bonds_mask = (bonds[:, 0] <= cut + self.crop_size - 1) & (
            bonds[:, 1] <= cut + self.crop_size - 1
        )
        bonds_mask &= (bonds[:, 0] >= cut) & (bonds[:, 1] >= cut)
        new_datum_["bonds"] = bonds[bonds_mask] - cut

        new_datum = MoleculeDatum(**new_datum_)
        return new_datum


class MoleculePad(MoleculeTransform):
    def __init__(self, pad_size: int):
        self.pad_size = pad_size

    def transform(self, datum: MoleculeDatum) -> MoleculeDatum:
        mol_size = datum.atom_token.shape[0]
        if mol_size >= self.pad_size:
            return datum

        new_datum_ = dict()
        for attr, obj in vars(datum).items():
            if type(obj) == np.ndarray:
                if attr == "bonds":
                    obj = pad_array(
                        obj, int(self.pad_size * self.pad_size // 2), value=-1
                    )
                elif attr in ["adjacency", "laplacian"]:
                    diff = self.pad_size - obj.shape[0]
                    obj = np.pad(obj, ((0, diff), (0, diff)))
                else:
                    obj = pad_array(obj, self.pad_size)
                new_datum_[attr] = obj
            else:
                new_datum_[attr] = obj

        new_datum = MoleculeDatum(**new_datum_)
        return new_datum


class Centralize(MoleculeTransform):
    def transform(self, datum):
        idxs = np.where(datum.atom_mask)
        datum.atom_coord[idxs] = datum.atom_coord[idxs] - datum.atom_coord[idxs].mean(
            axis=0
        )
        return datum


class CastToBFloat(MoleculeTransform):
    def transform(self, datum):
        new_datum_ = dict()
        for attr, obj in vars(datum).items():
            if type(obj) == np.ndarray and (obj.dtype in [np.float32, np.float64]):
                obj = obj.astype(jnp.bfloat16)
            new_datum_[attr] = obj
        return MoleculeDatum(**new_datum_)


class DescribeGraph(MoleculeTransform):
    def transform(self, datum: MoleculeDatum) -> MoleculeDatum:
        bonds = np.where((datum.bonds[..., -1] == -1)[..., None], 65, datum.bonds)
        adj = np.zeros((len(datum.atom_coord), len(datum.atom_coord)))
        adj[datum.bonds[:, 0], datum.bonds[:, 1]] = 1
        num_atoms = datum.atom_mask.sum()
        L = laplacian(adj[:num_atoms, :num_atoms], normed=False)
        _, chi = np.linalg.eigh(L)
        return MoleculeDatum(
            **vars(datum),
            adjacency=adj,
            laplacian=chi,
        )


class Permuter(MoleculeTransform):
    """Permute the atoms of a molecule and the bonds"""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def __call__(self, datum: MoleculeDatum) -> MoleculeDatum:
        np.random.seed(self.seed)
        permutation = np.random.permutation(len(datum.atom_token))
        permuted_bonds = np.zeros_like(datum.bonds)
        for i in range(datum.bonds.shape[0]):
            for j in range(datum.bonds.shape[1]):
                permuted_bonds[i, j] = permutation[datum.bonds[i, j]]

        return MoleculeDatum(
            **vars(datum),
            atom_token=datum.atom_token[permutation],
            atom_coord=datum.atom_coord[permutation],
            atom_mask=datum.atom_mask[permutation],
            bonds=permutation[datum.bonds],
        )
