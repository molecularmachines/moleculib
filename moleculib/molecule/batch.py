from __future__ import annotations

from functools import partial, reduce
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat

from .datum import MoleculeDatum
from .utils import pad_array


class MoleculeCollator:
    """
    Abstract class for collation of multiple ProteinDatum instances
    into a single batch instance
    """

    @classmethod
    def collate(cls, data_list: List[MoleculeDatum]):
        """
        Takes a list of ProteinDatum instances and produces
        a ProteinCollate instance with all the instances collated
        according to the appropriate collate function
        """
        raise NotImplementedError("collate method has to be implemented")

    def revert(self) -> List[MoleculeDatum]:
        """
        Reverts a batch to its original list form
        """
        raise NotImplementedError("revert method has to be implemented")

    def to(self, device: str) -> MoleculeCollator:
        for attr, obj in vars(self).items():
            if type(obj) == np.ndarray:
                setattr(self, attr, obj.to(device))
        return self

    def torch(self) -> MoleculeCollator:
        import torch

        for attr, obj in vars(self).items():
            if type(obj) == np.ndarray:
                setattr(self, attr, torch.from_numpy(obj))
        return self

    def to_dict(self, attrs=None):
        if attrs is None:
            attrs = vars(self).keys()
        dict_ = {}
        for attr in attrs:
            obj = getattr(self, attr)
            # strings are not JAX types
            if type(obj) in [list, tuple]:
                if type(obj[0]) not in [int, float]:
                    continue
                obj = jnp.array(obj)
            dict_[attr] = obj
        return dict_


class MoleculePadBatch(MoleculeCollator):
    def __init__(self, pad_mask, **kwargs):
        super().__init__()
        self.pad_mask = pad_mask
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def revert(self, constructor=MoleculeDatum):
        data_list = []
        for batch_index in range(self.pad_mask.shape[0]):
            mask = self.pad_mask[batch_index]
            data_attr = dict()
            for attr, obj in vars(self).items():
                if attr == "pad_mask":
                    continue
                obj = obj[batch_index]
                if type(obj) == np.ndarray:
                    obj = obj[mask]
                data_attr[attr] = obj
            data_list.append(constructor(**data_attr))
        return data_list

    @classmethod
    def collate(cls, data_list):
        proxy = data_list[0]
        data_type = type(proxy)
        unique_type = reduce(lambda res, obj: type(obj) is data_type, data_list, True)
        assert unique_type, "all data must have same type"
        max_size = max([len(datum.atom_token) for datum in data_list])

        def _maybe_pad_and_stack(obj_list):
            obj = obj_list[0]
            if type(obj) != np.ndarray:
                return obj_list
            new_list = map(partial(pad_array, total_size=max_size), obj_list)
            return np.stack(list(new_list), axis=0)

        keys = vars(proxy).keys()
        assert "bonds_list" not in keys, "PadBatch does not support bonds"
        value_lists = [vars(datum).values() for datum in data_list]
        value_lists = zip(*value_lists)
        values = list(map(_maybe_pad_and_stack, value_lists))
        batch_attr = dict(zip(keys, values))

        pad_mask = _maybe_pad_and_stack(
            [np.ones((len(datum.atom_token),)) for datum in data_list]
        ).astype(bool)

        return cls(pad_mask, **batch_attr)


class MoleculeGeometricBatch(MoleculeCollator):
    def __init__(self, batch_index, num_nodes, **kwargs):
        super().__init__()
        self.batch_index = batch_index
        self.num_nodes = num_nodes

        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def revert(self, constructor=MoleculeDatum):
        data_list = []
        for batch_index in range(len(self.num_nodes)):
            batch_mask = self.batch_index == batch_index
            data_attr = dict()
            for attr, obj in vars(self).items():
                if attr in ["batch_index", "num_nodes"]:
                    continue
                if type(obj) == np.ndarray:
                    obj = obj[batch_mask]
                else:
                    obj = obj[batch_index]
                data_attr[attr] = obj
            data_list.append(constructor(**data_attr))
        return data_list

    @classmethod
    def collate(cls, data_list):
        proxy = data_list[0]
        data_type = type(proxy)
        unique_type = reduce(lambda _, obj: type(obj) is data_type, data_list, True)
        assert unique_type, "all data must have same type"

        num_nodes = np.array([len(datum.residue_index) for datum in data_list])
        nodes_cumsum = np.cumsum([0] + list(num_nodes[:-1]))
        batch_index = [
            np.full((size), fill_value=idx) for idx, size in enumerate(num_nodes)
        ]
        batch_index = np.concatenate(batch_index, axis=0)

        def maybe_reindex(item):
            key, obj_list = item
            if key not in ["bonds_list", "angles_list", "dihedrals_list", "flips_list"]:
                return obj_list
            reindex_obj_list = [
                obj + reindex * 14 for (reindex, obj) in zip(nodes_cumsum, obj_list)
            ]
            return reindex_obj_list

        def maybe_concatenate(obj_list):
            obj = obj_list[0]
            if type(obj) != np.ndarray:
                return obj_list
            return np.concatenate(obj_list, axis=0)

        keys = vars(proxy).keys()
        value_lists = [vars(datum).values() for datum in data_list]
        value_lists = zip(*value_lists)
        value_lists = list(map(maybe_reindex, zip(keys, value_lists)))
        values = list(map(maybe_concatenate, value_lists))
        batch_attr = dict(zip(keys, values))

        return cls(batch_index=batch_index, num_nodes=num_nodes, **batch_attr)
