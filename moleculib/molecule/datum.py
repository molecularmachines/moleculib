import numpy as np
from .alphabet import PERIODIC_TABLE

import biotite.structure.io.mmtf as mmtf
import biotite.structure.io.mol as mol
from biotite.structure import Atom, array
from biotite.structure import BondList
import biotite.structure.io.pdb as pdb
from biotite.structure import connect_via_distances
from biotite.database import rcsb



class MoleculeDatum:
    
    def __init__(
        self,
        idcode: str,
        atom_token: np.ndarray,
        atom_coord: np.ndarray,
        atom_name: np.ndarray,
        atom_mask: np.ndarray,
        bonds: np.ndarray,
    ):
        self.idcode = idcode
        self.atom_token = atom_token
        self.atom_coord = atom_coord
        self.atom_name = atom_name
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
    def from_filepath(cls, filepath, ligand=None):
        pdb_file = mmtf.MMTFFile.read(filepath)
        atom_array = mmtf.get_structure(pdb_file, model=1)
        header = dict(
            idcode='allancomebackhere',
            resolution=None,
        )

        atom_array = atom_array[atom_array.hetero & (atom_array.res_name != 'HOH')]
        ligand_arrays = [] 
        ligand_names = np.unique(atom_array.res_name)
        for ligand_name in ligand_names:
            if ligand is not None and ligand_name != ligand:
                continue
            ligand_arrays.append(
                atom_array[atom_array.res_name == ligand_name]
            )
        
        return [
            cls.from_atom_array(atom_array, header=header)
            for atom_array in ligand_arrays
        ]

    @classmethod
    def fetch_pdb_id(cls, id, save_path=None, ligand=None):
        filepath = rcsb.fetch(id, "mmtf", save_path)
        return cls.from_filepath(filepath, ligand=ligand)

    @classmethod
    def from_atom_array(cls, atom_array, header):
        if atom_array.array_length() == 0:
            return cls.empty_molecule()

        atom_array.element[
            atom_array.element == "D"
        ] = "H"  # set deuterium to hydrogen (use mass_number to differentiate later)

        atom_token = np.array(
            [PERIODIC_TABLE.index(el) for el in atom_array.element]
        )

        atom_mask = np.ones(len(atom_token), dtype=bool)
        bonds = connect_via_distances(atom_array)

        return cls(
            idcode=header["idcode"],
            atom_token=atom_token,
            atom_coord=atom_array.coord,
            atom_name=atom_array.atom_name,
            atom_mask=atom_mask,
            bonds=bonds,
        )

    def to_atom_array(self):
        tokens = self.atom_token[self.atom_mask]
        coords = self.atom_coord[self.atom_mask]
        atoms = []
        for coord, token in zip(coords, tokens):
            el = PERIODIC_TABLE[token]
            atom = Atom(
                coord,
                chain_id="A",
                element=el,
                hetero=False,
                atom_name=el,
            )
            atoms.append(atom)
        arr = array(atoms)
        # bonds = BondList(len(atoms))
        # if self.bonds is not None:
            # print(self.bonds.shape)
            # bonds._bonds = self.bonds[self.bonds[:, 0] != -1]
        arr.bonds = self.bonds
        return arr

    def to_sdf_str(self):
        file = mol.MOLFile()
        file.set_structure(self.to_atom_array())
        return str(file)


    def plot(self, view, viewer=None, color=None):
        if viewer is None:
            viewer = (0, 0)
        view.addModel(self.to_sdf_str(), 'sdf', viewer=viewer)
        if color is not None:
            view.addStyle({'model': -1}, {'stick': {'color': color}, 'sphere': {'color': color, 'radius': 0.5}}, viewer=viewer)
        else:
            view.addStyle({'model': -1}, {'sphere': {'radius': 0.5},  'stick': {}}, viewer=viewer)
        return view
