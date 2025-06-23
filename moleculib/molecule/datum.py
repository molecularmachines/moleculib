import numpy as np
from flax import struct

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





# class PlotDensity:
#     def __init__(self, num_samples=None, window_size=(250, 250), pairs=False):
#         self.num_samples = num_samples
#         self.window_size = window_size
#         self.pairs = pairs

#     def __call__(self, run, batch, outputs=None, step=None):
#         # transform 9 outputs in 3x3 grid:
#         # Create figure
#         if self.num_samples is None:
#             self.num_samples = len(outputs)

#         fig = make_subplots(
#             rows=2,
#             cols=self.num_samples,
#             column_widths=[self.window_size[0]] * self.num_samples,
#             row_heights=[self.window_size[1]] * 2,
#             specs=[
#                 [{"type": "scene"}] * self.num_samples,
#                 [{"type": "scene"}] * self.num_samples,
#             ],
#         )
#         # loop over batch and outputs and plot density
#         for i, (output, ground) in enumerate(
#             list(zip(outputs[: self.num_samples], batch[: self.num_samples]))
#         ):
#             grid = ground.grid
#             atom_coord = ground.atom_coord
#             g = self.plot_density(grid, ground.density, atom_coord)
#             fig.add_trace(g[0], row=2, col=i + 1)
#             fig.add_trace(g[1], row=2, col=i + 1)
#             o = self.plot_density(grid, output, atom_coord)
#             fig.add_trace(o[0], row=1, col=i + 1)
#             fig.add_trace(o[1], row=1, col=i + 1)

#         run.log({"density": fig}, step=step)
#         return

#     def plot_density(self, grid, density, atom_coord):
#         X, Y, Z = grid[..., 0], grid[..., 1], grid[..., 2]
#         values = np.array(density)
#         levels = np.linspace(np.min(values), np.max(values), 20)

#         # Create isosurface plot
#         isosurface = go.Isosurface(
#             x=X.flatten(),
#             y=Y.flatten(),
#             z=Z.flatten(),
#             value=values.flatten(),
#             isomin=np.min(values),
#             isomax=np.max(values),
#             opacity=0.6,  # Adjust opacity for better visibility
#             surface_count=len(levels),  # Number of isosurfaces to show
#             caps=dict(
#                 x_show=False, y_show=False, z_show=False
#             ),  # Hide caps for better visualization
#             showscale=False,
#         )

#         # Create scatter plot for atoms
#         atoms_scatter = go.Scatter3d(
#             x=atom_coord[:, 0],
#             y=atom_coord[:, 1],
#             z=atom_coord[:, 2],
#             mode="markers",
#             marker=dict(
#                 size=10,
#             ),
#         )

#         return isosurface, atoms_scatter
