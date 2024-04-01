
from moleculib.molecule.datum import MoleculeDatum
from moleculib.protein.datum import ProteinDatum
from typing import List
from simple_pytree import Pytree


class AssemblyDatum(Pytree):

    def __init__(
            self, 
            protein_data: List[ProteinDatum] = None,
            molecule_data: List[MoleculeDatum] = None,
        ):
        self.protein_data = protein_data
        self.molecule_data =  molecule_data
    
    def from_datalist(datalist):
        protein_data = [datum for datum in datalist if datum.is_protein()]
        return AssemblyDatum(protein_data)

    def filter_proteins(self, keep=None, drop=None):
        """ indices to keep or drop """
        if keep is not None:
            new_protein_data = [self.protein_data[i] for i in keep]
        elif drop is not None:
            new_protein_data = [self.protein_data[i] for i in range(len(self.protein_data)) if i not in drop]
        else: 
            raise ValueError('Either keep or drop must be provided')
        return AssemblyDatum(new_protein_data)

    def plot(self, view, viewer=None, protein_style=dict(), molecule_style=dict()):
        if self.protein_data:
            for protein in self.protein_data:
                view = protein.plot(view, viewer, **protein_style)
        if self.molecule_data:
            for molecule in self.molecule_data:
                view = molecule.plot(view, viewer, **molecule_style)
        return view
