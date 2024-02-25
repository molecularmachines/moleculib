
from moleculib.protein.datum import ProteinDatum
from typing import List

class ComplexDatum:

    def __init__(self, protein_data: List[ProteinDatum]):
        self.protein_data = protein_data
    
    def from_datalist(datalist):
        protein_data = [datum for datum in datalist if datum.is_protein()]
        return ComplexDatum(protein_data)
