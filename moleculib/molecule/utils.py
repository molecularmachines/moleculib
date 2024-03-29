import os
from pathlib import Path

from biotite.structure import (
    filter_amino_acids,
    filter_monoatomic_ions,
    filter_nucleotides,
)

import numpy as np
import biotite.structure.io.mmtf as mmtf
import py3Dmol
import jax.numpy as jnp
import jaxlib
from jax.tree_util import register_pytree_node
import jax
from typing import List, Tuple, Any

home_dir = str(Path.home())
config = {"cache_dir": os.path.join(home_dir, ".cache", "moleculib")}

_solvent_list = ["DOD", "HOH", "SOL"]  # added DOD
_unknown_list = ["UNX", "UNL"]  # unknown atom or ion, or ligand


def pdb_to_atom_array(mmtf_file):
    atom_array = mmtf.get_structure(
        mmtf_file,
        model=1,
        extra_fields=["atom_id", "b_factor", "occupancy", "charge"],
        include_bonds=True,
    )
    # keep only molecular atoms
    atom_array = atom_array[~filter_amino_acids(atom_array)]
    atom_array = atom_array[~filter_nucleotides(atom_array)]
    atom_array = atom_array[~filter_monoatomic_ions(atom_array)]
    atom_array = atom_array[~np.isin(atom_array.res_name, _solvent_list)]
    atom_array = atom_array[
        ~np.isin(atom_array.res_name, _unknown_list)
    ]  # unknown atom or ion
    return atom_array


def pids_file_to_list(pids_path):
    with open(pids_path) as f:
        pids_str = f.read()
    return pids_str.rstrip().split(",")


def pad_array(array, total_size, value=0):
    shape = array.shape[1:]
    size = array.shape[0]
    diff = total_size - size
    assert diff >= 0
    if diff == 0:
        return array
    pad = np.full((diff, *shape), value, dtype=array.dtype)
    return np.concatenate((array, pad), axis=0)


def plot_py3dmol_grid(grid, window_size=(250, 250), spin=False):
    v = py3Dmol.view(
        viewergrid=(len(grid), len(grid[0])),
        linked=True,
        width=len(grid[0]) * window_size[0],
        height=len(grid) * window_size[1],
    )
    for i, row in enumerate(grid):
        for j, datum in enumerate(row):
            v.addModel(datum.to_sdf_str(), "sdf", viewer=(i, j))
            v.setStyle(
                {"sphere": {"radius": 0.4}, "stick": {"radius": 0.2}}, viewer=(i, j)
            )
    v.zoomTo()
    if spin:
        v.spin()
    v.setBackgroundColor("rgb(0,0,0)", 0)
    return v


def inner_stack(pytrees):
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values, axis=0), *pytrees)


def inner_split(pytree):
    leaves, defs = jax.tree_util.tree_flatten(pytree)
    splits = [
        [arr.squeeze(0) for arr in jnp.split(leaf, len(leaf), axis=0)]
        for leaf in leaves
    ]
    splits = list(zip(*splits))
    return [jax.tree_util.tree_unflatten(defs, split) for split in splits]


ACCEPTED_FORMATS = [
    np.ndarray,
    jnp.ndarray,
    jaxlib.xla_extension.ArrayImpl,
    jax.interpreters.partial_eval.DynamicJaxprTracer,
]

ACCEPTED_TYPES = [np.float64, np.float32, np.int64, np.int32, np.bool_]


def register_pytree(Datum):
    def encode_datum_pytree(datum: Datum) -> List[Tuple]:
        attrs = []
        went_through = False
        for attr, obj in vars(datum).items():
            # NOTE(Allan): come back here and make it universal
            if (type(obj) == object) or (
                (type(obj) in ACCEPTED_FORMATS) and (obj.dtype in ACCEPTED_TYPES)
            ):
                went_through = True
                attrs.append(obj)
            else:
                attrs.append(None)
        if not went_through:
            breakpoint()
        return attrs, vars(datum).keys()

    def decode_datum_pytree(keys, values: List[Any]) -> Datum:
        return Datum(**dict(zip(keys, values)))

    register_pytree_node(Datum, encode_datum_pytree, decode_datum_pytree)
