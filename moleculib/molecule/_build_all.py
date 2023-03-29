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

print(ds.metadata[ds.metadata.res_name=="SPM"].atom_count.unique())

filter = (ds.metadata.res_name=="SPM") & (ds.metadata.atom_count==20)
print(np.sum(filter))
print(ds.metadata[filter].idcode)

fil2 = ds.metadata.idcode == "1DPL"
print(ds.metadata[fil2])

fil2 = ds.metadata.idcode == "1M69"
print(ds.metadata[fil2])