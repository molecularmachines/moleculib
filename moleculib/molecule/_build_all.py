from moleculib import MoleculeDataset

path = "/mas/projects/molecularmachines/db/PDB"
# MoleculeDataset.build(save_path=path, max_workers=30)


ds = MoleculeDataset(path)  

print("Unique res: ", ds.metadata.res_name.nunique())
print("Unique res id: ", ds.metadata.res_id.nunique())