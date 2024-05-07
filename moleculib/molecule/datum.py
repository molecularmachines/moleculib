import numpy as np
from .alphabet import elements

import biotite.structure.io.mmtf as mmtf
import biotite.structure.io.mol as mol
from biotite.structure import Atom, array
from biotite.structure import BondList
from .utils import register_pytree


class MoleculeDatum:
    def __init__(
        self,
        atom_token: np.ndarray,
        atom_coord: np.ndarray,
        atom_mask: np.ndarray,
        bonds: np.ndarray,
        **kwargs,
    ):
        self.atom_token = atom_token
        self.atom_coord = atom_coord
        self.atom_mask = atom_mask
        self.bonds = bonds
        for k, v in kwargs.items():
            setattr(self, k, v)

    def empty(self):
        new_datum_ = dict()
        for attr, obj in vars(self).items():
            new_datum_[attr] = np.zeros_like(obj)
        return self.__class__(**new_datum_)

    def to_atom_array(self):
        tokens = self.atom_token
        coords = self.atom_coord
        atoms = []
        for coord, token in zip(coords, tokens):
            if token == 0:
                break
            el = elements.iloc[int(token) - 1]
            atom = Atom(
                np.array(coord),
                chain_id="A",
                element=el.symbol,
                hetero=False,
                atom_name=el.symbol,
            )
            atoms.append(atom)
        arr = array(atoms)
        bonds = BondList(len(atoms))
        if self.bonds is not None:
            sub = np.array(self.bonds[self.bonds[:, -1] != -1])
            pairs = np.sort(sub[:, :-1], axis=1)
            sub = np.unique(np.column_stack([pairs, sub[:, -1]]), axis=0)
            sorted_indices = np.lexsort((sub[:, 1], sub[:, 0]))
            bonds._bonds = sub[sorted_indices]
        arr.bonds = bonds
        return arr

    def to_sdf_str(self):
        file = mol.MOLFile()
        file.set_structure(self.to_atom_array())
        return str(file)


register_pytree(MoleculeDatum)


class QM9Datum(MoleculeDatum):
    def __init__(
        self,
        atom_token: np.ndarray,
        atom_coord: np.ndarray,
        atom_mask: np.ndarray,
        bonds: np.ndarray,
        properties: np.ndarray,
        stds: np.ndarray,
        **kwargs,
    ):
        super().__init__(atom_token, atom_coord, atom_mask, bonds, **kwargs)
        """
        Properties found in https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html
        """
        self.relevant_properties = {
            # 0: "μ Dipole moment, D",
            # 1: "α Isotropic polarizability, a₀³",
            2: "ε_HOMO Highest occupied molecular orbital energy, eV",
            3: "ε_LUMO Lowest unoccupied molecular orbital energy, eV",
            4: "Δε Gap between ε_HOMO and ε_LUMO, eV",
            # 5: "⟨R²⟩ Electronic spatial extent, a₀²",
            # 6: "ZPVE Zero point vibrational energy, eV",
            # 7: "U₀ Internal energy at 0K, eV",
            # 8: "U Internal energy at 298.15K, eV",
            # 9: "H Enthalpy at 298.15K, eV",
            # 10: "G Free energy at 298.15K, eV",
            # 11: "c_v Heat capacity at 298.15K, cal/(mol K)",
            # 12: "U₀_ATOM Atomization energy at 0K, eV",
            # 13: "U_ATOM Atomization energy at 298.15K, eV",
            # 14: "H_ATOM Atomization enthalpy at 298.15K, eV",
            # 15: "G_ATOM Atomization free energy at 298.15K, eV",
            # 16: "A Rotational constant, GHz",
            # 17: "B Rotational constant, GHz",
            # 18: "C Rotational constant, GHz",
        }
        self.properties = properties
        self.stds = stds
        # self.signature = np.bincount(self.atom_token, minlength=9 + 1)


register_pytree(QM9Datum)


class RSDatum(MoleculeDatum):
    def __init__(
        self,
        atom_token: np.ndarray,
        atom_coord: np.ndarray,
        atom_mask: np.ndarray,
        bonds: np.ndarray,
        properties: np.ndarray,
        adjacency: np.ndarray,
        laplacian: np.ndarray,
        **kwargs,
    ):
        super().__init__(atom_token, atom_coord, atom_mask, bonds, **kwargs)
        self.properties = properties
        self.adjacency = adjacency
        self.laplacian = laplacian


register_pytree(RSDatum)


from biotite.database import rcsb
from biotite.structure import get_molecule_masks
from .utils import pdb_to_atom_array


class PDBMoleculeDatum(MoleculeDatum):
    """
    Incorporates molecular data to MoleculeDatum
    """

    def __init__(
        self,
        idcode: str,
        resolution: float,
        chain_id: np.ndarray,
        res_id: np.ndarray,
        res_name: np.ndarray,
        atom_token: np.ndarray,
        atom_coord: np.ndarray,
        atom_name: np.ndarray,
        b_factor: np.ndarray,
        atom_mask: np.ndarray,
        bonds: np.ndarray,
    ):
        self.idcode = idcode
        self.resolution = resolution
        self.chain_id = chain_id
        self.res_id = res_id
        self.res_name = res_name
        self.atom_token = atom_token
        self.atom_coord = atom_coord
        self.atom_name = atom_name
        self.b_factor = b_factor
        self.atom_mask = atom_mask
        self.bonds = bonds

    def __len__(self):
        return len(self.sequence)

    @classmethod
    def empty_molecule(cls):
        return cls(
            idcode="",
            resolution=0.0,
            chain_id=np.array([]),
            res_id=np.array([]),
            res_name=np.array([]),
            atom_token=np.array([]),
            atom_coord=np.array([]),
            atom_name=np.array([]),
            b_factor=np.array([]),
            atom_mask=np.array([]),
            bonds=np.array([]),
        )

    @classmethod
    def from_filepath(cls, filepath, molecule_idx=None):
        if filepath.endswith(".mmtf"):
            mmtf_file = mmtf.MMTFFile.read(filepath)
            atom_array = pdb_to_atom_array(mmtf_file)
            header = dict(
                idcode=mmtf_file["structureId"],
                resolution=(
                    None if ("resolution" not in mmtf_file) else mmtf_file["resolution"]
                ),
            )
        elif filepath.endswith(".sdf"):
            mol_file = mol.MOLFile.read(filepath)
            atom_array = mol_file.get_structure()
            header = dict(idcode="allancomebackhere", resolution=None)
        else:
            raise NotImplementedError(
                f"File type {filepath.split('.')[-1]} is not supported"
            )
        return cls.from_atom_array(atom_array, header=header, molecule_idx=molecule_idx)

    @classmethod
    def fetch_pdb_id(cls, id, save_path=None):
        filepath = rcsb.fetch(id, "mmtf", save_path, verbose=False)
        return cls.from_filepath(filepath)

    @classmethod
    def from_atom_array(cls, atom_array, header, molecule_idx=None):
        if atom_array.array_length() == 0:
            return cls.empty_molecule()

        # (Ilan) to add other attributes from mendeleev for atoms
        # atom_attrs = ["spin", "mass_number",...] #other attributes from mendeleev
        # atom_extract = dict()
        # for attr in atom_attrs:
        #     atom_extract[attr] = elements.loc[orig_indexes][attr].to_numpy()

        atom_array.element[atom_array.element == "D"] = (
            "H"  # set deuterium to hydrogen (use mass_number to differentiate later)
        )

        try:
            orig_indexes = (
                elements.reset_index()
                .set_index("symbol")
                .loc[atom_array.element, "index"]
            )
        except KeyError as e:
            print(atom_array)
            print(e)
            raise e
        atom_token = elements.loc[orig_indexes]["atomic_number"].to_numpy()

        atom_mask = get_molecule_masks(atom_array)[0]
        bonds = atom_array.bonds._bonds

        if molecule_idx is not None:
            return cls(
                idcode=header["idcode"],
                resolution=header["resolution"],
                chain_id=atom_array.chain_id[atom_mask[molecule_idx]][0],
                res_id=atom_array.res_id[atom_mask[molecule_idx]][0],
                res_name=atom_array.res_name[atom_mask[molecule_idx]][0],
                atom_token=atom_token[atom_mask[molecule_idx]],
                atom_coord=atom_array.coord[atom_mask[molecule_idx]],
                atom_name=atom_array.atom_name[atom_mask[molecule_idx]],
                b_factor=atom_array.b_factor[atom_mask[molecule_idx]],
                atom_mask=np.full(np.sum(atom_mask[molecule_idx]), 1),
                bonds=atom_array.bonds[
                    atom_mask[molecule_idx]
                ].bond_type_matrix(),  # BondType
            )
        return cls(
            idcode=header["idcode"],
            resolution=header["resolution"],
            chain_id=atom_array.chain_id,
            res_id=atom_array.res_id,
            res_name=atom_array.res_name,
            atom_token=atom_token,
            atom_coord=atom_array.coord,
            atom_name=atom_array.atom_name,
            b_factor=atom_array.b_factor if hasattr(atom_array, "b_factor") else None,
            atom_mask=atom_mask,
            bonds=bonds,
        )

    def to_atom_array(self):
        tokens = self.atom_token[self.atom_mask]
        coords = self.atom_coord[self.atom_mask]
        atoms = []
        for coord, token in zip(coords, tokens):
            el = elements.iloc[token]
            atom = Atom(
                coord,
                chain_id="A",
                element=el.symbol.upper(),
                hetero=False,
                atom_name=el.symbol.upper(),
            )
            atoms.append(atom)
        arr = array(atoms)
        bonds = BondList(len(atoms))
        if self.bonds is not None:
            bonds._bonds = self.bonds[self.bonds[:, 0] != -1]
        arr.bonds = bonds
        return arr

    def to_sdf_str(self):
        file = mol.MOLFile()
        file.set_structure(self.to_atom_array())
        return str(file)


import os


class CrossdockDatum(MoleculeDatum):
    AA_NAME_SYM = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
    }

    AA_NAME_NUMBER = {k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())}

    AA_NUMBER_NAME = {v: k for k, v in AA_NAME_NUMBER.items()}

    def __init__(self, *args, **kwargs):
        self.key = kwargs.pop("key")
        self.filename = kwargs.pop("filename")
        self.atom_features = kwargs.pop("atom_features")
        self.protein_token = kwargs.pop("protein_token")
        self.protein_coord = kwargs.pop("protein_coord")
        self.protein_mask = kwargs.pop("protein_mask")
        self.base_path = "/mas/projects/molecularmachines/db/crossdocked_targetdiff/crossdocked_v1.1_rmsd1.0/"

        super().__init__(*args, **kwargs)

    def coord_center(self):
        return (self.atom_coord * self.atom_mask[:, None]).sum(0) / self.atom_mask.sum()

    def protein_pdb_str(self):
        folder, file = self.filename.split("/")
        pdb_path = os.path.join(
            self.base_path, folder, "_".join(file.split("_")[:3]) + ".pdb"
        )
        with open(pdb_path, "r") as f:
            s = str(f.read())
        return s


register_pytree(CrossdockDatum)


class PDBBindDatum(MoleculeDatum):
    def __init__(self, *args, **kwargs):
        self.pdb_id = kwargs.pop("pdb_id")
        try:
            self.pka = kwargs.pop("pka")
        except KeyError:
            self.pka = None
        self.charge = kwargs.pop("charge")
        self.protein_token = kwargs.pop("protein_token")
        self.protein_coord = kwargs.pop("protein_coord")
        self.protein_mask = kwargs.pop("protein_mask")
        self.base_path = "/mas/projects/molecularmachines/db/PDBBind/refined-set"

        super().__init__(*args, **kwargs)

    def coord_center(self):
        return (self.atom_coord * self.atom_mask[:, None]).sum(0) / self.atom_mask.sum()

    def protein_pdb_str(self):
        pdb_path = os.path.join(
            self.base_path, f"{self.pdb_id}", f"{self.pdb_id}_protein.pdb"
        )
        with open(pdb_path, "r") as f:
            s = str(f.read())
        return s

    def pocket_pdb_str(self):
        pdb_path = os.path.join(
            self.base_path, f"{self.pdb_id}", f"{self.pdb_id}_pocket.pdb"
        )
        with open(pdb_path, "r") as f:
            s = str(f.read())
        return s


register_pytree(PDBBindDatum)


class DensityDatum(MoleculeDatum):
    def __init__(self, *args, **kwargs):
        self.density = kwargs.pop("density")
        self.grid = kwargs.pop("grid")

        super().__init__(*args, **kwargs)


register_pytree(DensityDatum)

from moleculib.molecule.h5_to_pdb import create_pdb
from copy import deepcopy

class MISATODatum(MoleculeDatum):
    def __init__(self, *args, **kwargs):
        self.pdb_id = kwargs.pop("pdb_id")
        self.protein_token = kwargs.pop("protein_token")
        self.protein_coord = kwargs.pop("protein_coord")
        self.protein_mask = kwargs.pop("protein_mask")
        self.atoms_residue = kwargs.pop("atoms_residue")
        self.atoms_type = kwargs.pop("atoms_type")

        super().__init__(*args, **kwargs)


    def replace(self, **kwargs):
        _vars = deepcopy(vars(self))
        _vars.update(**kwargs)
        return self.__class__(**_vars)

    def at(self, i):
        return self.__class__(
            atom_token=self.atom_token,
            atom_coord=self.atom_coord[..., i, :, :],
            atom_mask=self.atom_mask,
            bonds=self.bonds,
            pdb_id=self.pdb_id,
            protein_token=self.protein_token,
            protein_coord=self.protein_coord[..., i, :, :],
            protein_mask=self.protein_mask,
            atoms_residue=self.atoms_residue,
            atoms_type=self.atoms_type,
        )

    def neighborhood_idxs(self, r):
        return np.where(
            (
                ((self.atom_coord[0][None] - self.protein_coord[0][:, None]) ** 2).sum(
                    -1
                )
                ** 0.5
                < r
            ).sum(1)
        )[0]

    def keep_neighborhood(self, r):
        idxs = self.neighborhood_idxs(r)
        return self.__class__(
            atom_token=self.atom_token,
            atom_coord=self.atom_coord,
            atom_mask=self.atom_mask,
            bonds=self.bonds,
            pdb_id=self.pdb_id,
            protein_token=self.protein_token[idxs],
            protein_coord=self.protein_coord[..., :, idxs, :],
            protein_mask=self.protein_mask[idxs],
            atoms_residue=self.atoms_residue[idxs],
            atoms_type=self.atoms_type[idxs],
        )

    def dehydrate(self):
        h_atom = np.where(self.atom_token != 1)[0]
        h_protein = np.where(self.protein_token != 1)[0]
        return self.__class__(
            atom_token=self.atom_token[h_atom],
            atom_coord=self.atom_coord[..., h_atom, :],
            atom_mask=self.atom_mask[h_atom],
            bonds=self.bonds,
            pdb_id=self.pdb_id,
            protein_token=self.protein_token[h_protein],
            protein_coord=self.protein_coord[..., h_protein, :],
            protein_mask=self.protein_mask[h_protein],
            atoms_residue=self.atoms_residue[h_protein],
            atoms_type=self.atoms_type[h_protein],
        )

    def pdb_str(self, frame):
        return "\n".join(
            create_pdb(
                self.protein_coord[frame],
                self.atoms_type,
                self.protein_token,
                self.atoms_residue,
                [],
            )
        )


register_pytree(MISATODatum)
