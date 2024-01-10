from .datum import MoleculeDatum, QM9Datum
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

        new_datum = datum.__class__(**new_datum_)
        return new_datum


class MoleculePad(MoleculeTransform):
    def __init__(self, pad_size: int):
        self.pad_size = pad_size

    def transform(self, datum: MoleculeDatum) -> MoleculeDatum:
        mol_size = datum.atom_token.shape[0]
        if mol_size > self.pad_size:  # TODO: make sure not >=
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
                elif attr in ["properties", "stds"]:
                    pass
                else:
                    obj = pad_array(obj, self.pad_size)
            new_datum_[attr] = obj

        new_datum = datum.__class__(**new_datum_)
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
        return datum.__class__(**new_datum_)


class DescribeGraph(MoleculeTransform):
    def transform(self, datum: QM9Datum) -> QM9Datum:
        adj = np.zeros((len(datum.atom_coord), len(datum.atom_coord)))
        adj[datum.bonds[:, 0], datum.bonds[:, 1]] = 1
        num_atoms = datum.atom_mask.sum()
        L = laplacian(adj[:num_atoms, :num_atoms], normed=False)
        _, chi = np.linalg.eigh(L)
        return datum.__class__(
            **vars(datum),
            adjacency=adj,
            laplacian=chi,
        )


class Permuter(MoleculeTransform):
    """Permute the atoms of a molecule and the bonds"""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def transform(self, datum: MoleculeDatum) -> MoleculeDatum:
        np.random.seed(self.seed)
        permutation = np.random.permutation(len(datum.atom_token))
        permuted_bonds = np.zeros_like(datum.bonds)
        for i in range(datum.bonds.shape[0]):
            for j in range(datum.bonds.shape[1]):
                permuted_bonds[i, j] = permutation[datum.bonds[i, j]]
        raise NotImplementedError("bond perm must be corrected")
        # return datum.__class__(
        #     **vars(datum),
        #     atom_token=datum.atom_token[permutation],
        #     atom_coord=datum.atom_coord[permutation],
        #     atom_mask=datum.atom_mask[permutation],
        #     bonds=permutation[datum.bonds],
        # )


from moleculib.molecule.alphabet import elements


class AtomFeatures(MoleculeTransform):
    """Compute atomistic features from the atom tokens"""

    def __init__(self):
        self.relevant_features = [
            "atomic_radius",
            "atomic_volume",
            "density",
            "dipole_polarizability",
            "electron_affinity",
            # "electronic_configuration", complex object
            "evaporation_heat",
            "fusion_heat",
            # "group_id", make onehot
            "lattice_constant",
            # "lattice_structure", complex object
            # "period", make onehot
            # "series_id", make onehot ?
            "specific_heat_capacity",
            "thermal_conductivity",
            "vdw_radius",
            "covalent_radius_cordero",
            "covalent_radius_pyykko",
            "en_pauling",
            "en_allen",
            "proton_affinity",
            "gas_basicity",
            "heat_of_formation",
            "c6",
            "covalent_radius_bragg",
            "vdw_radius_bondi",
            "vdw_radius_truhlar",
            "vdw_radius_rt",
            "vdw_radius_batsanov",
            "vdw_radius_dreiding",
            "vdw_radius_uff",
            "vdw_radius_mm3",
            # "abundance_crust", relevant? maybe in an evolutionary manner..
            # "abundance_sea",
            "en_ghosh",
            "vdw_radius_alvarez",
            "c6_gb",
            "atomic_weight",
            "atomic_weight_uncertainty",
            # "is_monoisotopic",
            # "is_radioactive",
            "atomic_radius_rahm",
            # "geochemical_class",
            # "goldschmidt_class",
            "metallic_radius",
            "metallic_radius_c12",
            "covalent_radius_pyykko_double",
            "covalent_radius_pyykko_triple",
            "dipole_polarizability_unc",
            "pettifor_number",
            "glawe_number",
            "molar_heat_capacity",
        ]
        self.params = {}
        for f in self.relevant_features:
            v = elements[f].dropna().values
            self.params[f] = (v.min(), v.max(), v.mean(), v.std())

    def shift(self, x, f):
        # make x between [-1,1]
        mn, mx, _, _ = self.params[f]
        m = (1 - (-1)) / (mx - mn)
        b = (-1) - m * mn
        x = m * x + b
        return np.nan_to_num(x, nan=-2.0)  # TODO: -2 for missing, find better solution

    def transform(self, datum: QM9Datum) -> QM9Datum:
        atom_types = datum.atom_token[datum.atom_mask] - 1
        atom_features = -3 * np.ones(
            (datum.atom_token.shape[0], len(self.relevant_features))
        )  # TODO: -3 for masks, find better solution
        atoms = elements.iloc[atom_types]
        for i, feature in enumerate(self.relevant_features):
            atom_features[datum.atom_mask, i] = self.shift(
                atoms[feature].values, feature
            )

        return datum.__class__(**vars(datum), atom_features=atom_features)

class NormalizeProperties(MoleculeTransform):
    def __init__(self):
        self.mins, self.maxs, self.means, self.stds = (
            np.array([ 0.00000000e+00,  1.29899998e+01, -1.16627998e+01, -4.76199245e+00,
                    1.88302791e+00,  3.53641014e+01,  4.34048831e-01, -1.69097949e+04,
                    -1.69095625e+04, -1.69095371e+04, -1.69107148e+04,  6.27799988e+00,
                    -1.13110725e+02, -1.13889809e+02, -1.14609604e+02, -1.04810410e+02,
                    0.00000000e+00,  3.37119997e-01,  3.31180006e-01], dtype=np.float32),
            np.array([ 2.29605007e+01,  1.30860001e+02, -2.76739788e+00,  5.26540327e+00,
                    1.69282036e+01,  3.28602026e+03,  7.43942928e+00, -1.10148779e+03,
                    -1.10140979e+03, -1.10138403e+03, -1.10202295e+03,  4.63810005e+01,
                    -1.30881882e+01, -1.31352901e+01, -1.31866665e+01, -1.25200958e+01,
                    2.32663781e+05,  1.57709976e+02,  1.57706985e+02], dtype=np.float32),
            np.array([ 2.6704957e+00,  7.5256264e+01, -6.5378528e+00,  3.1930840e-01,
                    6.8571572e+00,  1.1888519e+03,  4.0535598e+00, -1.1182805e+04,
                    -1.1182573e+04, -1.1182548e+04, -1.1183713e+04,  3.1614559e+01,
                    -7.6084610e+01, -7.6548729e+01, -7.6986252e+01, -7.0808083e+01,
                    1.0557292e+01,  1.4054270e+00,  1.1276687e+00], dtype=np.float32),
            np.array([1.50735915e+00, 8.17282867e+00, 5.99983990e-01, 1.27825534e+00,
                    1.28357017e+00, 2.79095245e+02, 9.03440654e-01, 1.09038257e+03,
                    1.09037683e+03, 1.09037683e+03, 1.09039624e+03, 4.06875277e+00,
                    1.03288488e+01, 1.04204378e+01, 1.04946375e+01, 9.50233650e+00,
                    1.28643958e+03, 1.01723623e+00, 9.62196529e-01], dtype=np.float32)
        )

    def transform(self, datum: QM9Datum) -> QM9Datum:
        new_datum_ = dict()
        for attr, obj in vars(datum).items():
            if attr == "properties":
                obj = (obj - self.means) / self.stds
            if attr == "stds":
                obj = self.stds
            new_datum_[attr] = obj
        return datum.__class__(**new_datum_)
