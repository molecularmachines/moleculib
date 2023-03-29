from moleculib import MoleculeDataset
import numpy as np
import matplotlib.pyplot as plt
path = "/mas/projects/molecularmachines/db/PDB"
# MoleculeDataset.build(save_path=path, max_workers=30)


ds = MoleculeDataset(path)  
a = ds.metadata.atom_count.hist(bins=1022)
count, division = np.histogram(ds.metadata.atom_count, bins=1023)
print("Unique res: ", ds.metadata.res_name.nunique())
print("Unique res id: ", ds.metadata.res_id.nunique())