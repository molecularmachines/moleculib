from moleculib.protein.datum import ProteinDatum
from moleculib.molecule.datum import MoleculeDatum
import py3Dmol
import os
from typing import Callable


def plot_py3dmol_grid(
        grid, 
        window_size=(250, 250), 
        sphere=False,
        ribbon=False,
    ):
    v = py3Dmol.view(
        viewergrid=(len(grid), len(grid[0])),
        linked=True,
        width=len(grid[0]) * window_size[0],
        height=len(grid) * window_size[1],
    )
    for i, row in enumerate(grid):
        for j, datum in enumerate(row):
            datum.plot(v, viewer=(i, j), sphere=sphere, ribbon=ribbon)
    v.zoomTo()
    v.setBackgroundColor("rgb(0,0,0)", 0)
    return v


def plot_py3dmol_molecule(
        grid, 
        window_size=(250, 250), 
        sphere=False,
        ribbon=False,
    ):
    v = py3Dmol.view(
        viewergrid=(len(grid), len(grid[0])), 
        linked=True, 
        width=len(grid[0]) * window_size[0],  
        height=len(grid) * window_size[1],
    )
    for i, row in enumerate(grid):
        for j, datum in enumerate(row):
            v.addModel(datum.to_sdf_str(), 'sdf', viewer=(i, j))
            v.setStyle({'sphere': {'radius': 0.4, 'color': 'orange'}, 'stick': {'radius': 0.2, 'color': 'orange'} }, viewer=(i, j))
    v.zoomTo()
    v.setBackgroundColor('rgb(0,0,0)', 0)
    return v


from tempfile import gettempdir
import wandb
import time
import numpy as np


class PlotPy3DmolSamples:
    def __init__(
        self, plot_func: Callable, num_samples=7, window_size=(250, 250), pairs=False
    ):
        self.plot_func = plot_func
        self.num_samples = num_samples
        self.window_size = window_size
        self.pairs = pairs

    def __call__(self, run, outputs, batch):
        # transform 9 outputs in 3x3 grid:
        outputs = outputs[: self.num_samples]
        batch = batch[: self.num_samples]
        if self.pairs:
            datums = []
            for output, ground in list(zip(outputs, batch)):
                datum = MoleculeDatum(
                    idcode=None,
                    atom_token=ground.atom_token,
                    atom_coord=np.array(output.coord),
                    atom_mask=ground.atom_mask,
                    bonds=None,
                )
                datums.append(datum)
            v = self.plot_func([datums, batch], window_size=self.window_size)
        else:
            v = self.plot_func(outputs, window_size=self.window_size)
        html = v._make_html()

        html_path = os.path.join(gettempdir(), f"{run.name}.html")
        with open(html_path, "w") as f:
            f.write(html)
        run.log({"samples": wandb.Html(open(html_path))})

        time.sleep(1)
        os.remove(html_path)
