import numpy as np
from Bio.PDB import parse_pdb_header
from biotite.structure import get_molecule_masks
from biotite.database import rcsb
from mendeleev.fetch import fetch_table
from .utils import pdb_to_atom_array


class MoleculeDatum:
    """
    Incorporates molecular data to MoleculeDatum
    """

    def __init__(
        self,
        idcode: str,
        resolution: float,
        chain_id: np.ndarray,
        res_name: np.ndarray,
        atom_token: np.ndarray,
        atom_coord: np.ndarray,
        atom_name: np.ndarray,
        molecule_mask: np.ndarray,
    ):
        self.idcode = idcode
        self.resolution = resolution
        self.chain_id = chain_id
        self.res_name = res_name
        self.atom_token = atom_token
        self.atom_coord = atom_coord
        self.atom_name = atom_name
        self.molecule_mask = molecule_mask

    def __len__(self):
        return len(self.sequence)

    @classmethod
    def empty_molecule(cls):
        return cls(
            idcode="",
            resolution=0.0,
            chain_id=np.array([]),
            res_name=np.array([]),
            atom_token=np.array([]),
            atom_coord=np.array([]),
            atom_name=np.array([]),
            molecule_mask=np.array([]),
        )

    @classmethod
    def from_filepath(cls, filepath, molecule_idx=None):
        atom_array = pdb_to_atom_array(filepath)
        header = parse_pdb_header(filepath)
        return cls.from_atom_array(atom_array, header=header, molecule_idx=molecule_idx)

    @classmethod
    def fetch_pdb_id(cls, id, save_path=None):
        filepath = rcsb.fetch(id, "pdb", save_path)
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

        elements = fetch_table("elements").assign(
            symbol=lambda df: df.symbol.str.upper()
        )
        orig_indexes = (
            elements.reset_index().set_index("symbol").loc[atom_array.element, "index"]
        )
        atom_token = elements.loc[orig_indexes]["atomic_number"].to_numpy()

        molecule_mask = get_molecule_masks(atom_array)
        if molecule_idx is not None:
            return cls(
                idcode=header["idcode"],
                resolution=header["resolution"],
                chain_id=atom_array.chain_id[molecule_mask[molecule_idx]][0],
                res_name=atom_array.res_name[molecule_mask[molecule_idx]][0],
                atom_token=atom_token[molecule_mask[molecule_idx]],
                atom_coord=atom_array.coord[molecule_mask[molecule_idx]],
                atom_name=atom_array.atom_name[molecule_mask[molecule_idx]],
                molecule_mask=np.full(np.sum(molecule_mask[molecule_idx]), True),
            )
        return cls(
            idcode=header["idcode"],
            resolution=header["resolution"],
            chain_id=atom_array.chain_id,
            res_name=atom_array.res_name,
            atom_token=atom_token,
            atom_coord=atom_array.coord,
            atom_name=atom_array.atom_name,
            molecule_mask=molecule_mask,
        )
