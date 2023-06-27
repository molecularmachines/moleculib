from typing import List, Callable


class MetricsPipe:
    def __init__(self, metrics_list: List[Callable]):
        self.metrics_list = metrics_list

    def __call__(self, datum):
        metrics = {}
        for metric in self.metrics_list:
            metrics.update(metric(datum))
        return metrics
