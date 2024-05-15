import os
from typing import Callable
from tempfile import gettempdir
import wandb
import time
import numpy as np
from copy import deepcopy
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class PlotPy3DmolSamples:
    def __init__(
        self, plot_func: Callable, num_samples=None, window_size=(250, 250), pairs=False
    ):
        self.plot_func = plot_func
        self.num_samples = num_samples
        self.window_size = window_size
        self.pairs = pairs

    def __call__(self, run, batch, outputs=None, step=None):
        # transform 9 outputs in 3x3 grid:
        if self.num_samples is not None:
            batch = batch[: self.num_samples]
            if outputs is not None:
                outputs = outputs[: self.num_samples]
        if outputs is not None:
            if self.pairs:
                datums = []
                for output, ground in list(zip(outputs, batch)):
                    datum = deepcopy(ground)
                    datum.atom_token = np.array(output.token).astype(np.int32)
                    datum.atom_coord = np.array(output.coord + output.coord_center)
                    datum.atom_mask = np.array(output.mask).astype(np.int32)
                    datum.fixed_atoms = np.array(output.fixed).astype(np.int32)
                    ground.fixed_atoms = np.array(output.fixed).astype(np.int32)
                    datum.bonds = None
                    datums.append(datum)
                v = self.plot_func([datums, batch], window_size=self.window_size)
            else:
                v = self.plot_func(outputs, window_size=self.window_size)
        else:
            v = self.plot_func([batch], window_size=self.window_size)

        html = v._make_html()

        html_path = os.path.join(gettempdir(), f"{run.name}.html")
        with open(html_path, "w") as f:
            f.write(html)
        run.log({"samples": wandb.Html(open(html_path))}, step=step)

        time.sleep(1)
        os.remove(html_path)
        return


# class PlotWeights:
#     def __init__(
#         self, plot_func: Callable, window_size=(250, 250)
#     ):
#         self.plot_func = plot_func
#         self.window_size = window_size

#     def __call__(self, run, batch, outputs=None, step=None):
#         # transform 9 outputs in 3x3 grid:
#         if self.num_samples is not None:
#             batch = batch[: self.num_samples]
#             if outputs is not None:
#                 outputs = outputs[: self.num_samples]
#         if outputs is not None:
#             if self.pairs:
#                 datums = []
#                 for output, ground in list(zip(outputs, batch)):
#                     datum = MoleculeDatum(
#                         idcode=None,
#                         atom_token=ground.atom_token,
#                         atom_coord=np.array(output.coord),
#                         atom_mask=ground.atom_mask,
#                         bonds=None,
#                     )
#                     datums.append(datum)
#                 v = self.plot_func([datums, batch], window_size=self.window_size)
#             else:
#                 v = self.plot_func(outputs, window_size=self.window_size)
#         else:
#             v = self.plot_func([batch], window_size=self.window_size)

#         html = v._make_html()

#         html_path = os.path.join(gettempdir(), f"{run.name}.html")
#         with open(html_path, "w") as f:
#             f.write(html)
#         run.log({"samples": wandb.Html(open(html_path))}, step=step)

#         time.sleep(1)
#         os.remove(html_path)
#         return


class PlotDensity:
    def __init__(self, num_samples=None, window_size=(250, 250), pairs=False):
        self.num_samples = num_samples
        self.window_size = window_size
        self.pairs = pairs

    def __call__(self, run, batch, outputs=None, step=None):
        # transform 9 outputs in 3x3 grid:
        # Create figure
        if self.num_samples is None:
            self.num_samples = len(outputs)

        fig = make_subplots(
            rows=2,
            cols=self.num_samples,
            column_widths=[self.window_size[0]] * self.num_samples,
            row_heights=[self.window_size[1]] * 2,
            specs=[
                [{"type": "scene"}] * self.num_samples,
                [{"type": "scene"}] * self.num_samples,
            ],
        )
        # loop over batch and outputs and plot density
        for i, (output, ground) in enumerate(
            list(zip(outputs[: self.num_samples], batch[: self.num_samples]))
        ):
            grid = ground.grid
            atom_coord = ground.atom_coord
            g = self.plot_density(grid, ground.density, atom_coord)
            fig.add_trace(g[0], row=2, col=i + 1)
            fig.add_trace(g[1], row=2, col=i + 1)
            o = self.plot_density(grid, output, atom_coord)
            fig.add_trace(o[0], row=1, col=i + 1)
            fig.add_trace(o[1], row=1, col=i + 1)

        run.log({"density": fig}, step=step)
        return

    def plot_density(self, grid, density, atom_coord):
        X, Y, Z = grid[..., 0], grid[..., 1], grid[..., 2]
        values = np.array(density)
        levels = np.linspace(np.min(values), np.max(values), 20)

        # Create isosurface plot
        isosurface = go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            isomin=np.min(values),
            isomax=np.max(values),
            opacity=0.6,  # Adjust opacity for better visibility
            surface_count=len(levels),  # Number of isosurfaces to show
            caps=dict(
                x_show=False, y_show=False, z_show=False
            ),  # Hide caps for better visualization
            showscale=False,
        )

        # Create scatter plot for atoms
        atoms_scatter = go.Scatter3d(
            x=atom_coord[:, 0],
            y=atom_coord[:, 1],
            z=atom_coord[:, 2],
            mode="markers",
            marker=dict(
                size=10,
            ),
        )

        return isosurface, atoms_scatter
