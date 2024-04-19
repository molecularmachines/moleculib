


from collections import defaultdict
from typing import List

from moleculib.protein.transform import DescribeChemistry

from ..protein.metrics import ProteinMetric
from .datum import AssemblyDatum
from copy import deepcopy



class AssemblyMetric:

    def transform(self, datum: AssemblyDatum) -> AssemblyDatum:
        """
        Takes as input an individual data point, processes
        the values in it and returns a new ProteinDatum
        """
        raise NotImplementedError("method transform must be implemented")


class ApplyMetricsToProtein(AssemblyMetric):
    
    def __init__(self, protein_metrics: List[ProteinMetric]):
        self.protein_metrics = protein_metrics

    def __call__(self, datum):
        protein_data = datum.protein_data
        protein_data = [DescribeChemistry().transform(protein) for protein in protein_data]
        metrics = defaultdict(list)
        for metric in self.protein_metrics:
            for protein in protein_data:
                metrics_protein = metric(protein)
                for key, value in metrics_protein.items():
                    metrics[key].append(value)
        metrics = {key: sum(value) / len(value) for key, value in metrics.items()}
        return metrics
