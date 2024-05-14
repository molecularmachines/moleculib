from moleculib.protein.datum import ProteinDatum
from moleculib.molecule.datum import MoleculeDatum
from moleculib.nucleic.datum import NucleicDatum

import py3Dmol
import os
from typing import Callable
from colour import Color

DEFAULT_COLORS = [
    "cyan",
    "orange",
    "lime",
]


def plot_py3dmol(
        data, 
        color: str = "DEFAULT",
        **kwargs,
    ):
    v = py3Dmol.view()
    if color == "DEFAULT":
        colors = [ DEFAULT_COLORS[i] for i in range(len(data))]
    else:
        # make a gradient from red to green
        colors = list(Color('green').range_to(Color('red'), len(data)))
        colors = [c.get_hex_l() for c in colors]

    for i, datum in enumerate(data):
        datum.plot(v, color=colors[i], **kwargs)
    v.zoomTo()
    v.setBackgroundColor("rgb(0,0,0)", 0)
    return v


def plot_py3dmol_grid(
        grid, 
        window_size=(250, 250), 
        **kwargs
    ):
    v = py3Dmol.view(
        viewergrid=(len(grid), len(grid[0])),
        linked=True,
        width=len(grid[0]) * window_size[0],
        height=len(grid) * window_size[1],
    )
    for i, row in enumerate(grid):
        for j, datum in enumerate(row):
            if type(datum) == list:
                plot_py3dmol_traj(datum, v, viewer=(i, j), **kwargs)
            else:
                datum.plot(v, viewer=(i, j), **kwargs)
    v.zoomTo()
    v.setBackgroundColor("rgb(0,0,0)", 0)
    return v


def traj_to_pdb(traj, num_steps=100, head=0, tail=5):
    models = ""
    if len(traj) > num_steps:
        downsample = len(traj) // num_steps
        traj = traj[::downsample]
    traj = traj + [traj[-1]] * tail
    for i, p in enumerate(traj):
        models += f"MODEL {i + 1}\n"
        models += p.to_pdb_str()
        models += "\nENDMDL\n"
    return models

def plot_py3dmol_traj(traj, view, viewer, num_steps=100):
    models = traj_to_pdb(traj, num_steps=num_steps)
    view.addModelsAsFrames(models, 'pdb', viewer=viewer)
    view.setStyle({}, viewer=viewer)
    view.addStyle({'atom': 'CA'}, {'sphere': {'radius': 0.4, 'color': 'darkgray'}}, viewer=viewer)
    # view.addStyle({'chain': 'A'}, {'sphere': {'radius': 0.4, 'color': 'darkgray'}}, viewer=viewer)
    view.addStyle({'chain': 'A'}, {'stick': {}, 'cartoon': {'color': 'spectrum'}}, viewer=viewer)
    
    view.setBackgroundColor("rgb(0,0,0)", 0, viewer=viewer)
    view.animate({'loop': 'forward'}, viewer=viewer)
    return view
