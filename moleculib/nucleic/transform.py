#need to implement transform which will 
#be able to crop and pad the nucleic datum
import os
from einops import repeat, rearrange
import math
import biotite
from moleculib.nucleic.datum import NucleicDatum,dna_res_tokens, rna_res_tokens
from moleculib.nucleic.alphabet import * 
#UNK_TOKEN
import numpy as np
from moleculib.nucleic.utils import pad_array,pids_file_to_list

import jax.numpy as jnp
from tqdm import tqdm


#followed Allan but not sure exactly why we need all these abstraction:

class NucTransform:
    """
    Abstract class for nucleic transform
    """
    def transform(self, datum: NucleicDatum) -> NucleicDatum:
        """
        Takes as input an individual data point, processes
        the values in it and returns a new ProteinDatum
        """
        raise NotImplementedError("method transform must be implemented")

class NucCrop(NucTransform):
    def __init__(self, crop_len):
        #also unsure why we need to init
        self.crop_len = crop_len
    
    def transform(self, datum):
        seq_len = len(datum)
        if seq_len<=self.crop_len:
            return datum
        
        start_index =  np.random.randint(low=0, high=(seq_len - self.crop_len))
        new_datum_ = dict()
        for attr, obj in vars(datum).items():
            if type(obj) in [np.ndarray, list, tuple, str] and len(obj) == seq_len:
                new_datum_[attr] = obj[start_index : start_index + self.crop_len]
            else:
                new_datum_[attr] = obj

        new_datum = type(datum)(**new_datum_)
        return new_datum

class NucPad(NucTransform):
    def __init__(self, pad_size):
        self.pad_size = pad_size
    
    def transform(self, datum):
        seq_len = len(datum)
        #QUESTION: So we ignore all nucs???
        if seq_len >= self.pad_size:
            datum.pad_mask = np.ones_like(datum.residue_token)
            return datum
        
        size_diff = self.pad_size - seq_len
        shift = np.random.randint(0, size_diff)

        new_datum_ = dict()
        for attr, obj in vars(datum).items():
            if type(obj) == np.ndarray and attr != "label" and len(obj) == seq_len:
                obj = pad_array(obj, self.pad_size) #func from utils
                obj = np.roll(obj, shift, axis=0)
                new_datum_[attr] = obj
            else:
                new_datum_[attr] = obj
        pad_mask = pad_array(np.ones_like(datum.nuc_token), self.pad_size)
        new_datum_["pad_mask"] = pad_mask
        new_datum = type(datum)(**new_datum_)
        return new_datum


