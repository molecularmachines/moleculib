from moleculib.protein.datum import ProteinDatum
from moleculib.molecule.datum import MoleculeDatum
import py3Dmol
import os
from typing import Callable

from tempfile import gettempdir
import wandb
import time
import numpy as np
from copy import deepcopy


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
