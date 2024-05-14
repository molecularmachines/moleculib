


from collections import defaultdict
from typing import List

from moleculib.abstract.metrics import MetricsPipe
from moleculib.protein.transform import DescribeChemistry

from ..protein.metrics import ProteinMetric
from .datum import AssemblyDatum
from copy import deepcopy

import inspect


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

    def __call__(self, datum, maybe_other_datum=None):
        protein_data = datum.protein_data
        protein_data = [DescribeChemistry().transform(protein) for protein in protein_data]
        metrics = defaultdict(list)
        pipe = MetricsPipe(self.protein_metrics)
        if maybe_other_datum is not None:
            for protein, protein_other in zip(protein_data, maybe_other_datum.protein_data):
                metrics_ = pipe(protein, protein_other)
                for key, value in metrics_.items():
                    metrics[key].append(value)
        else:
            for protein in protein_data:
                metrics_ = pipe(protein)
                for key, value in metrics_.items():
                    metrics[key].append(value)
        for key, value in metrics.items():
            metrics[key] = sum(value) / len(value)                
        return metrics
