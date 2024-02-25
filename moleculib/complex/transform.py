from .datum import ComplexDatum
from ..protein.datum import ProteinDatum
from typing import List
import numpy as np 



class ComplexTransform:
    """
    Abstract class for transformation of ProteinDatum datapoints
    """

    def transform(self, datum: ComplexDatum) -> ComplexDatum:
        """
        Takes as input an individual data point, processes
        the values in it and returns a new ProteinDatum
        """
        raise NotImplementedError("method transform must be implemented")


from copy import deepcopy
from moleculib.protein.transform import ProteinTransform

class ApplyToProteins(ComplexTransform):
    
    def __init__(self, protein_transform: List[ProteinTransform]):
        self.protein_transform = protein_transform

    def transform(self, datum):
        protein_data = deepcopy(datum.protein_data)
        new_protein_data = []
        for protein in protein_data:
            for transform in self.protein_transform:
                protein = transform.transform(protein)
            new_protein_data.append(protein)
        return ComplexDatum(new_protein_data)
            

class ComplexPad(ComplexTransform):

    def __init__(self, num_chains):
        self.num_chains = num_chains

    def transform(self, datum):
        num_chains = len(datum.protein_data)

        if num_chains < self.num_chains:
            protein_data = deepcopy(datum.protein_data)
            for _ in range(self.num_chains - num_chains):
                protein_data.append(ProteinDatum.empty())
            return ComplexDatum(protein_data)
        else:
            return datum


class FilterProteinChains(ComplexTransform):

    def __init__(self, num_chains):
        self.num_chains = num_chains

    def transform(self, datum):
        protein_data = datum.protein_data

        def protein_distance(protein1, protein2):
            coord1 = protein1.atom_coord[..., 1, :]
            coord2 = protein2.atom_coord[..., 1, :]
            mask1 = protein1.atom_mask[..., 1]
            mask2 = protein2.atom_mask[..., 1]

            vector_map = coord1[:, None] - coord2
            map_mask = mask1[:, None] | mask2
            
            distance_map = np.linalg.norm(vector_map, axis=-1)
            distance_map = np.ma.array(distance_map, mask=~map_mask)
            min_distance = distance_map.min()
            return min_distance

        distance_map = {
            i: [ protein_distance(protein, protein2) for protein2 in protein_data ]
            for i, protein in enumerate(protein_data)
        }

        protein_data = deepcopy(datum.protein_data)

        remaining_indices = set(range(len(protein_data)))
        new_protein_data = []
        acceptable = []

        while len(new_protein_data) < min(self.num_chains, len(protein_data)):
            if not acceptable:
                acceptable = list(remaining_indices)
            index = np.random.choice(acceptable)

            protein = protein_data[index]
            new_protein_data.append(protein)
            remaining_indices.remove(index)

            distances = distance_map[index]

            acceptable = [ 
                i for i, distance in enumerate(distances) if (distance < 10) and (i != index) and (i in remaining_indices) ]

        return ComplexDatum(new_protein_data)            
