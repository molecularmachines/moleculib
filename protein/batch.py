from functools import reduce, partial
import numpy as np
from .datum import ProteinDatum
from typing import List


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

        return cls(batch_index, num_nodes, **batch_attr)
