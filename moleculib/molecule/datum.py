from flax import struct
import numpy as np

from .alphabet import PERIODIC_TABLE


@struct.dataclass
class MoleculeDatum:
    """
    Incorporates protein data to MolecularDatum
    and reshapes atom arrays to residue-based representation
    """

    idcode: str

    atom_type: np.ndarray
    atom_coord: np.ndarray
    atom_mask: np.ndarray
    # atom_radius: np.ndarray = None

    # bonds_list: np.ndarray = None
    # bonds_mask: np.ndarray = None
    # angles_list: np.ndarray = None
    # angles_mask: np.ndarray = None
    # dihedrals_list: np.ndarray = None
    # dihedrals_mask: np.ndarray = None

    @classmethod
    def from_atom_array(cls, atom_array):
        return cls(
            idcode=None,
            atom_type=np.array(
                [
                    PERIODIC_TABLE.index(el)
                    for el in atom_array.get_annotation("element")
                ]
            ),
            atom_coord=atom_array._coord,
            atom_mask=np.ones(len(atom_array), dtype=bool),
        )

    def to_pdb_str(self):
        """
        Converts the atom array to a PDB string representation.
        """
        lines = []

        names = [PERIODIC_TABLE[i] for i in self.atom_type]
        res_name = "MOL"
        res_index = 0

        for idx, (name, coord) in enumerate(zip(names, self.atom_coord)):
            x, y, z = coord
            line = list(" " * 80)
            line[0:6] = "ATOM".ljust(6)
            line[6:11] = str(idx + 1).ljust(5)
            line[12:16] = name.ljust(4)
            line[17:20] = res_name.ljust(3)
            line[21:22] = "A"
            line[23:27] = str(res_index + 1).ljust(4)
            line[30:38] = f"{x:.3f}".rjust(8)
            line[38:46] = f"{y:.3f}".rjust(8)
            line[46:54] = f"{z:.3f}".rjust(8)
            line[76:78] = name[0].rjust(2)
            lines.append("".join(line))

        return "\n".join(lines)

    # def to_atom_array(self):
    #     """
    #     Converts the atom array to a PDB string representation.
    #     """
    #     atom_array = AtomArray(
    #         idcode = self.idcode,
    #         atom_type = self.atom_type,
    #         atom_coord = self.atom_coord,
    #         atom_mask = self.atom_mask,
    #     )
    #     return atom_array

    def __repr__(self):
        return f"MoleculeDatum(shape={self.atom_type.shape})"

    def plot(self, view=None):
        pass
