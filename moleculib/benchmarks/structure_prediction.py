from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
import torch
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
from scipy.special import softmax
import gc


def esmfold_sequence(sequence,num_recycles=2):
    #Fix the sequence
    sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
    sequence = re.sub(":+",":",sequence)
    sequence = re.sub("^[:]+","",sequence)
    sequence = re.sub("[:]+$","",sequence)
    
    seed = "default"
    model = torch.load(os.environ["ESMFOLD_MODEL_PATH"])
    model.eval().cuda().requires_grad_(False)
    model_name_ = model_name
    mask_rate = 0.0
    model.train(False)
    
    output = model.infer(sequence,
                       num_recycles=num_recycles,
                       chain_linker="X"*chain_linker,
                       residue_index_offset=512,
                       mask_rate=mask_rate,
                       return_contacts=get_LM_contacts)
    
    pdb_str = model.output_to_pdb(output)[0]
    output = tree_map(lambda x: x.cpu().numpy(), output)
    plddt = output["plddt"][0,:,1].mean()
    return pdb_str,plddt

