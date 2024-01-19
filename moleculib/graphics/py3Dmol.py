from moleculib.protein.datum import ProteinDatum
from moleculib.molecule.datum import MoleculeDatum
import py3Dmol


def plot_py3dmol_protein_grid(
        grid, 
        window_size=(250, 250), 
        sphere=False,
        ribbon=False,
    ):
    v = py3Dmol.view(
        viewergrid=(len(grid), len(grid[0])), 
        linked=True, 
        width=len(grid[0]) * window_size[0],  
        height=len(grid) * window_size[1],
    )
    for i, row in enumerate(grid):
        for j, datum in enumerate(row):
            v.addModel(datum.to_pdb_str(), 'pdb', viewer=(i, j))
            if sphere:
                v.setStyle({'sphere': {'radius': 0.3}}, viewer=(i, j))
            elif ribbon:
                v.setStyle({'cartoon': {'color': 'spectrum', 'ribbon': True, 'thickness': 0.7}}, viewer=(i, j))
            else:
                v.setStyle({'cartoon': {'color': 'spectrum'}, 'stick': {'radius': 0.2}}, viewer=(i, j))
    v.zoomTo()
    v.setBackgroundColor('rgb(0,0,0)', 0)
    return v


def plot_py3dmol_molecule(
        grid, 
        window_size=(250, 250), 
        sphere=False,
        ribbon=False,
    ):
    v = py3Dmol.view(
        viewergrid=(len(grid), len(grid[0])), 
        linked=True, 
        width=len(grid[0]) * window_size[0],  
        height=len(grid) * window_size[1],
    )
    for i, row in enumerate(grid):
        for j, datum in enumerate(row):
            v.addModel(datum.to_sdf_str(), 'sdf', viewer=(i, j))
            v.setStyle({'sphere': {'radius': 0.4, 'color': 'orange'}, 'stick': {'radius': 0.2, 'color': 'orange'} }, viewer=(i, j))
    v.zoomTo()
    v.setBackgroundColor('rgb(0,0,0)', 0)
    return v