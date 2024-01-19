from moleculib import MoleculeDataset
import numpy as np
import plotly.express as px

path = "/mas/projects/molecularmachines/db/PDB"
# MoleculeDataset.build(save_path=path, max_workers=30)
#

ds = MoleculeDataset(path)
fig = px.histogram(ds.metadata.atom_count)
fig.show()
print("Unique res: ", ds.metadata.res_name.nunique())
print("Unique res id: ", ds.metadata.res_id.nunique())

print(ds.metadata[ds.metadata.res_name == "SPM"].atom_count.unique())

filter = (ds.metadata.res_name == "SPM") & (ds.metadata.atom_count == 20)
print(np.sum(filter))
print(ds.metadata[filter].idcode)

fil2 = ds.metadata.idcode == "1DPL"
print(ds.metadata[fil2])

fil2 = ds.metadata.idcode == "1M69"
print(ds.metadata[fil2])

fil3 = ds.metadata.atom_count == 4
print(ds.metadata[fil3].res_name.unique())
