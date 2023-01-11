from __future__ import annotations

import torch
import numpy as np
import jax.numpy as jnp
from functools import reduce, partial
from .datum import ProteinDatum
from typing import List, Tuple
from einops import repeat, rearrange
from .alphabet import backbone_atoms



class ProteinCollator:
    """
    Abstract class for collation of multiple ProteinDatum instances
    into a single batch instance
    """

    @classmethod
    def collate(cls, data_list: List[ProteinDatum]):
        """
        Takes a list of ProteinDatum instances and produces
        a ProteinCollate instance with all the instances collated
        according to the appropriate collate function
        """
        raise NotImplementedError("collate method has to be implemented")

    def revert(self) -> List[ProteinDatum]:
        """
        Reverts a batch to its original list form
        """
        raise NotImplementedError("revert method has to be implemented")

    def to(self, device: str) -> ProteinCollator:
        for attr, obj in vars(self).items():
            if type(obj) == np.ndarray:
                setattr(self, attr, obj.to(device))
        return self

    def torch(self) -> ProteinCollator:
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


def pad_array(array, total_size):
    shape = array.shape[1:]
    size = len(array)
    diff = total_size - size
    assert diff >= 0
    if diff == 0:
        return array

    pad = np.zeros((diff, *shape), dtype=array.dtype)
    return np.concatenate((array, pad), axis=0)


class PadBatch(ProteinCollator):
    def __init__(self, pad_mask, **kwargs):
        super().__init__()
        self.pad_mask = pad_mask
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def revert(self, constructor=ProteinDatum):
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
        max_size = max([len(datum.sequence) for datum in data_list])

        def _maybe_pad_and_stack(obj_list):
            obj = obj_list[0]
            if type(obj) != np.ndarray:
                return obj_list
            new_list = map(partial(pad_array, total_size=max_size), obj_list)
            return np.stack(list(new_list), axis=0)

        keys = vars(proxy).keys()
        value_lists = [vars(datum).values() for datum in data_list]
        value_lists = zip(*value_lists)
        values = list(map(_maybe_pad_and_stack, value_lists))
        batch_attr = dict(zip(keys, values))

        pad_mask = _maybe_pad_and_stack(
            [np.ones((len(datum.sequence),)) for datum in data_list]
        ).astype(bool)

        return cls(pad_mask, **batch_attr)


class GeometricBatch(ProteinCollator):
    def __init__(self, batch_index, num_nodes, **kwargs):
        super().__init__()
        self.batch_index = batch_index
        self.num_nodes = num_nodes
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def revert(self, constructor=ProteinDatum):
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
        unique_type = reduce(lambda res, obj: type(obj) is data_type, data_list, True)
        assert unique_type, "all data must have same type"

        def maybe_concatenate(obj_list):
            obj = obj_list[0]
            if type(obj) != np.ndarray:
                return obj_list
            return np.concatenate(obj_list, axis=0)

        keys = vars(proxy).keys()
        value_lists = [vars(datum).values() for datum in data_list]
        value_lists = zip(*value_lists)
        values = list(map(maybe_concatenate, value_lists))
        batch_attr = dict(zip(keys, values))

        num_nodes = [len(datum.sequence) for datum in data_list]
        batch_index = [
            np.full((size), fill_value=idx) for idx, size in enumerate(num_nodes)
        ]
        batch_index = np.concatenate(batch_index, axis=0)

        return cls(batch_index=batch_index, num_nodes=num_nodes, **batch_attr)


def complete_graph(batch_num_nodes) -> np.array:
    cumsum = np.cumsum([0] + list(batch_num_nodes[:-1]))
    batch_edge_2d = [
        np.stack(
            (
                repeat(np.arange(0, num_nodes), "i -> i j", j=num_nodes),
                repeat(np.arange(0, num_nodes), "j -> i j", i=num_nodes),
            ),
            axis=0,
        )
        for num_nodes in batch_num_nodes
    ]
    batch_edge_flattened = [
        rearrange(edges, "... i j -> ... (i j)") for edges in batch_edge_2d
    ]
    batch_edge_reindex = list(map(sum, zip(batch_edge_flattened, cumsum)))
    edge_index = np.concatenate(batch_edge_reindex, axis=-1)
    return edge_index


def radial_graph(
    coords, batch_num_nodes, max_radius: 15.0
) -> Tuple[np.array, np.array]:
    edge_index = (v, u) = complete_graph(batch_num_nodes)
    distances = np.linalg.norm(coords[v] - coords[u], axis=-1, keepdims=True)
    accepted = (distances < max_radius).squeeze(-1)
    return edge_index[:, accepted], distances[accepted]


class FullyConnectedGeometricBatch(GeometricBatch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.edge_index = complete_graph(self.num_nodes)


class RadialGeometricBatch(GeometricBatch):
    def __init__(self, max_radius, **kwargs) -> None:
        super().__init__(**kwargs)
        ca_coord = self.atom_coord[:, backbone_atoms.index("CA")]
        edge_index, _ = radial_graph(ca_coord, self.num_nodes, max_radius=max_radius)
        self.edge_index = edge_index
