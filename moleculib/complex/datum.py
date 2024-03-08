
from moleculib.protein.datum import ProteinDatum
from typing import List
from simple_pytree import Pytree


class ComplexDatum(Pytree):

    def __init__(self, protein_data: List[ProteinDatum]):
        self.protein_data = protein_data
    
    def from_datalist(datalist):
        protein_data = [datum for datum in datalist if datum.is_protein()]
        return ComplexDatum(protein_data)

    def plot(self, view, viewer=None, sphere=False, ribbon=False):
        for protein in self.protein_data:
            view = protein.plot(view, viewer, sphere, ribbon)
        return view

