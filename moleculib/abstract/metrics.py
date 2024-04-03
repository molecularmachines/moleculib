


from typing import Callable, List


class MetricsPipe:
    
    def __init__(self, metrics_list: List[Callable]):
        self.metrics_list = metrics_list

    def __call__(self, datum):
        if type(datum) == list:
            metrics_batch = [self.__call__(datum_) for datum_ in datum]
            metrics_keys = metrics_batch[0].keys()
            metrics = {}
            for key in metrics_keys:
                metrics[key] = sum([metric[key] for metric in metrics_batch]) / len(metrics_batch)
            return metrics
        else:
            metrics = {}
            for metric in self.metrics_list:
                metrics.update(metric(datum))
            return metrics
