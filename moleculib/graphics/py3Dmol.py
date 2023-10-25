from moleculib.protein.datum import ProteinDatum
from moleculib.molecule.datum import MoleculeDatum
import py3Dmol


def plot_py3dmol_grid(grid, window_size=(250, 250), spin=False):
    v = py3Dmol.view(
        viewergrid=(len(grid), len(grid[0])), 
        linked=True, 
        width=len(grid[0]) * window_size[0],  
        height=len(grid) * window_size[1],
    )
    for i, row in enumerate(grid):
        for j, datum in enumerate(row):
            if type(datum) == ProteinDatum:
                v.addModel(datum.to_pdb_str(), 'pdb', viewer=(i, j))
                v.setStyle({'cartoon': {'color': 'spectrum'}, 'stick': {'radius': 0.2}}, viewer=(i, j))
            elif type(datum) == MoleculeDatum:
                v.addModel(datum.to_sdf_str(), 'sdf', viewer=(i, j))
                v.setStyle({'sphere': {'radius': 0.4, 'color': 'orange'}, 'stick': {'radius': 0.2, 'color': 'orange'} }, viewer=(i, j))
    v.zoomTo()
    if spin: v.spin()
    v.setBackgroundColor('rgb(0,0,0)', 0)
    return v
