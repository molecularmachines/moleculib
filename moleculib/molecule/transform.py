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
                elif attr == "properties":
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
    def transform(self, datum: MoleculeDatum) -> MoleculeDatum:
        bonds = np.where((datum.bonds[..., -1] == -1)[..., None], 65, datum.bonds)
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

    def transform(self, datum: MoleculeDatum) -> MoleculeDatum:
        tokens = datum.atom_token
        atom_types = tokens[datum.atom_mask] - 1
        atom_features = -2 * np.ones(
            (len(tokens), len(self.relevant_features))
        )  #TODO: -2 for masks, find better solution
        atoms = elements.iloc[atom_types]
        for i, feature in enumerate(self.relevant_features):
            atom_features[datum.atom_mask, i] = (
                atoms[feature].fillna(-1).values
            )  #TODO: -1 for missing, find better solution

        return datum.__class__(**vars(datum), atom_features=atom_features)
