from .datum import AssemblyDatum 
from ..protein.datum import ProteinDatum
from typing import List
import numpy as np 

from copy import deepcopy
from moleculib.protein.transform import ProteinTransform


class AssemblyTransform:
    """
    Abstract class for transformation of ProteinDatum datapoints
    """

    def transform(self, datum: AssemblyDatum) -> AssemblyDatum:
        """
        Takes as input an individual data point, processes
        the values in it and returns a new ProteinDatum
        """
        raise NotImplementedError("method transform must be implemented")


class ApplyToProteins(AssemblyTransform):
    
    def __init__(self, protein_transform: List[ProteinTransform]):
        self.protein_transform = protein_transform

    def transform(self, datum):
        protein_data = deepcopy(datum.protein_data)
        new_protein_data = []
        for protein in protein_data:
            for transform in self.protein_transform:
                protein = transform.transform(protein)
            new_protein_data.append(protein)
        return AssemblyDatum(new_protein_data)
            

class ComplexPad(AssemblyTransform):

    def __init__(self, num_chains):
        self.num_chains = num_chains

    def transform(self, datum):
        num_chains = len(datum.protein_data)

        if num_chains < self.num_chains:
            protein_data = deepcopy(datum.protein_data)
            for _ in range(self.num_chains - num_chains):
                protein_data.append(ProteinDatum.empty())
            return AssemblyDatum(protein_data)
        else:
            return datum


class FilterProteinChains(AssemblyTransform):

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

        return AssemblyDatum(new_protein_data)            



class StackProteins(AssemblyTransform):

    def transform(self, datum):
        protein_data = datum.protein_data
        sample_datum = protein_data[0]
        attrs = dict()

        for attr, _ in vars(sample_datum).items():
            batched = []
            for protein in protein_data:
                batched.append(getattr(protein, attr))
            if isinstance(batched[0], np.ndarray):
                attrs[attr] = np.stack(batched, axis=0)
            else:
                attrs[attr] = np.empty((len(batched), ))

        return AssemblyDatum(protein_data=type(sample_datum)(**attrs))


class UnstackProteins(AssemblyTransform):

    def transform(self, datum):
        protein_data = datum.protein_data
        num_prot = len(protein_data.residue_token)

        attr_lists = {}
        for attr, _ in vars(protein_data).items():
            for i in range(num_prot):
                if attr not in attr_lists:
                    attr_lists[attr] = []
                if getattr(protein_data, attr) is None:
                    attr_lists[attr].append(None)
                else:
                    attr_lists[attr].append(getattr(protein_data, attr)[i])

        new_protein_data = []
        for i in range(num_prot):
            attrs = { attr: values[i] for attr, values in attr_lists.items() }
            new_protein_data.append(type(protein_data)(**attrs))
        
        return AssemblyDatum(new_protein_data)