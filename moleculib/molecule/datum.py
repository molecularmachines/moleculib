import numpy as np
from Bio.PDB import parse_pdb_header
from biotite.structure import get_molecule_masks
from biotite.database import rcsb
from .utils import pdb_to_atom_array
from .alphabet import elements

import biotite.structure.io.mmtf as mmtf
import biotite.structure.io.mol as mol
from biotite.structure import Atom, array 


class MoleculeDatum:
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
                resolution=None
                if ("resolution" not in mmtf_file)
                else mmtf_file["resolution"],
            )
        elif filepath.endswith('.sdf'):
            mol_file = mol.MOLFile.read(filepath)
            atom_array = mol_file.get_structure()
            header = dict(
                idcode='allancomebackhere',
                resolution=None
            )
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

        atom_array.element[
            atom_array.element == "D"
        ] = "H"  # set deuterium to hydrogen (use mass_number to differentiate later)

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

        atom_mask = get_molecule_masks(atom_array)

        # filter out molecules with 1 atom (which are not really molecules nor ions)
        atom_mask = atom_mask[np.sum(atom_mask, axis=1) > 1]
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
        tokens = self.atom_token[self.atom_mask[0]]
        atoms = []
        for (coord, (_, el)) in zip(
            self.atom_coord, 
            elements.iloc[tokens].iterrows()
        ): 
            atom = Atom(
                coord, 
                chain_id='A',
                element=el.symbol, 
                hetero=False, 
                atom_name=el.symbol
            )
            atoms.append(atom)
        arr = array(atoms)
        arr.bonds = self.bonds
        return arr

    def to_sdf_str(self):
        file = mol.MOLFile()
        file.set_structure(self.to_atom_array())
        return str(file)
