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
import e3nn_jax as e3nn
import jaxlib



#followed Allan but not sure exactly why we need all these abstraction:

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
        if seq_len<=self.crop_size:
            return datum
        if cut is None:
            cut = 0 #np.random.randint(low=0, high=(seq_len - self.crop_size))
        
        if type(datum) == list:
            return [self.transform(datum_, cut=cut) for datum_ in datum]
        
        new_datum_ = dict()
        for attr, obj in vars(datum).items():
            if attr == 'contact_map' and obj is not None:
                #Crop the contact map
                new_datum_[attr] = obj[cut : cut + self.crop_size, cut : cut + self.crop_size]
            elif attr == 'fmtoks' and obj is not None:
                new_datum_[attr] = obj[cut : cut + self.crop_size, :]
            elif attr == 'msa' and obj is not None:
                new_datum_[attr] = obj[:, cut : cut + self.crop_size]
            elif attr == 'attention' and obj is not None:
                new_datum_[attr] = obj[..., cut : cut + self.crop_size, cut : cut + self.crop_size]

            elif type(obj) in [np.ndarray, list, tuple, str, e3nn.IrrepsArray, jaxlib.xla_extension.ArrayImpl] and len(obj) == seq_len:
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
            datum.pad_mask = np.ones_like(datum.nuc_token)
            return datum
        
        size_diff = self.pad_size - seq_len
        # shift = np.random.randint(0, size_diff)

        new_datum_ = dict()
        for attr, obj in vars(datum).items():
            if attr == 'contact_map' and obj is not None:
                #Pad the contact map
                padded_contact_map = np.zeros((self.pad_size, self.pad_size), dtype=int)
                padded_contact_map[:seq_len, :seq_len] = obj[:, :]
                new_datum_[attr] = padded_contact_map
            
            elif attr == 'fmtoks' and obj is not None:
                #Pad the fmtoks
                padded_fmtoks = np.zeros((self.pad_size, obj.shape[1]), dtype=float)
                padded_fmtoks[:seq_len, :] = obj[:, :]
                new_datum_[attr] = padded_fmtoks
            
            elif attr == 'msa' and obj is not None:
                #Pad the msa
                depth = obj.shape[0]
                pad_msa = np.full((depth, self.pad_size), 12, dtype=float) #pad token is 12 for MSA 
                pad_msa[:, :seq_len] = obj[:, :seq_len]
                new_datum_[attr] = pad_msa
            elif attr == 'attention' and obj is not None:
                padded_attention = np.zeros((12, 20, self.pad_size, self.pad_size))
                padded_attention[..., :seq_len, :seq_len] = obj[..., :seq_len, :seq_len]
                new_datum_[attr] = padded_attention

            elif type(obj) == np.ndarray and attr != "label" and len(obj) == seq_len:
                obj = pad_array(obj, self.pad_size) #func from utils
                # obj = np.roll(obj, shift, axis=0)
                new_datum_[attr] = obj
            else:
                new_datum_[attr] = obj


        pad_mask = pad_array(np.ones_like(datum.nuc_token), self.pad_size)
        new_datum_["pad_mask"] = pad_mask

        new_datum = type(datum)(**new_datum_)
        
        return new_datum




class PrepareForPipeline(NucTransform):
    def __init__(self, msa_depth):
        self.msa_depth = msa_depth

    def transform(self, datum):
        # NOTE(Dana): Have fun
        new_datum_ = dict()
        for attr, obj in vars(datum).items():

            if attr == 'msa' and obj is not None:
                # new_datum_['msa'] = np.array(datum.msa)
                msa = np.array(datum.msa)
                current_depth = msa.shape[0]

                if current_depth < self.msa_depth:
                    # Pad with sequences of 12s
                    pad_sequences = np.full((self.msa_depth - current_depth, msa.shape[1]), 12)
                    msa = np.vstack([msa, pad_sequences])
                elif current_depth > self.msa_depth:
                    # Crop to desired depth vertically
                    msa = msa[:self.msa_depth, :]

                new_datum_['msa'] = msa
            
            elif attr in ['idcode', 'sequence', 'resolution']:
                new_datum_[attr] = None
            # elif attr == 'attention' and obj is not None:
            #     new_datum_["attention"] = None
            else:
                new_datum_[attr] = obj


        new_datum = type(datum)(**new_datum_)        
        return new_datum



