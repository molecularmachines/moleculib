# need to implement transform which will
# be able to crop and pad the nucleic datum
import math
import os

import biotite
import e3nn_jax as e3nn
import jax.numpy as jnp
import jaxlib
# UNK_TOKEN
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm

from moleculib.nucleic.alphabet import *
from moleculib.nucleic.datum import (NucleicDatum)
from moleculib.nucleic.utils import pad_array, pids_file_to_list

# followed Allan but not sure exactly why we need all these abstraction:


class NucTransform:
    """
    Abstract class for nucleic transform
    """

    def transform(self, datum: NucleicDatum) -> NucleicDatum:
        """
        Takes as input an individual data point, processes
        the values in it and returns a new NucleicDatum
        """
        raise NotImplementedError("method transform must be implemented")


class NucCrop(NucTransform):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def transform(self, datum, cut=None):
        seq_len = len(datum)
        if seq_len <= self.crop_size:
            return datum
        if cut is None:
            cut = 0  # np.random.randint(low=0, high=(seq_len - self.crop_size))

        if type(datum) == list:
            return [self.transform(datum_, cut=cut) for datum_ in datum]

        new_datum_ = dict()
        for attr, obj in vars(datum).items():
            if (
                type(obj)
                in [
                    np.ndarray,
                    list,
                    tuple,
                    str,
                    e3nn.IrrepsArray,
                    jaxlib.xla_extension.ArrayImpl,
                ]
                and len(obj) == seq_len
            ):
                new_datum_[attr] = obj[cut : cut + self.crop_size]
            else:
                new_datum_[attr] = obj

        new_datum = type(datum)(**new_datum_)

        return new_datum


class NucPad(NucTransform):
    def __init__(self, pad_size):
        self.pad_size = pad_size

    def transform(self, datum):
        # print(f'transform NucPad,  datum has msa: {datum.msa}')
        if type(datum) == list:
            return [self.transform(datum_) for datum_ in datum]

        seq_len = len(datum)

        if seq_len >= self.pad_size:
            # print("here because pad size is {self.pad_size}")
            # datum.pad_mask = np.ones_like(datum.nuc_token)
            new_datum = vars(datum)
            new_datum["pad_mask"] = np.ones_like(datum.nuc_token)
            return type(datum)(**new_datum)

        size_diff = self.pad_size - seq_len

        new_datum_ = dict()
        for attr, obj in vars(datum).items():
            if type(obj) == np.ndarray and attr != "label" and len(obj) == seq_len:
                obj = pad_array(obj, self.pad_size)  # func from utils
                new_datum_[attr] = obj
            else:
                new_datum_[attr] = obj

        pad_mask = np.zeros((self.pad_size,), dtype=bool)
        pad_mask[:seq_len] = True  # True for the original sequence, False for padding
        new_datum_["pad_mask"] = pad_mask
        new_datum = type(datum)(**new_datum_)

        return new_datum


class PrepareForPipeline(NucTransform):
    def __init__(self, msa_depth):
        self.msa_depth = msa_depth

    def transform(self, datum):
        new_datum_ = dict()
        for attr, obj in vars(datum).items():

            if attr == "msa":  # and obj is not None
                # new_datum_['msa'] = np.array(datum.msa)[0]
                msa = np.array(datum.msa)
                current_depth = msa.shape[0]

                if current_depth < self.msa_depth:
                    # Pad with sequences of 12s
                    pad_sequences = np.full(
                        (self.msa_depth - current_depth, msa.shape[1]), 12
                    )
                    msa = np.vstack([msa, pad_sequences])
                elif current_depth > self.msa_depth:
                    # Crop to desired depth vertically
                    msa = msa[: self.msa_depth, :]

                # Convert T token (2) to U token (1) over all MSAs:
                msa[msa == 2] = 1

                new_datum_["msa"] = msa

            elif attr in ["idcode", "sequence", "resolution"]:
                new_datum_[attr] = None
            # elif attr == 'attention' and obj is not None:
            #     new_datum_["attention"] = None
            else:
                new_datum_[attr] = obj

        new_datum = type(datum)(**new_datum_)
        return new_datum
