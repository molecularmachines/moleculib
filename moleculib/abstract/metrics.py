


from typing import Callable, List
from inspect import signature


class MetricsPipe:
    
    def __init__(self, metrics_list: List[Callable]):
        self.metrics_list = metrics_list

    def __call__(self, datum, maybe_other_datum=None):
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
                num_params = len(signature(metric).parameters)
                if num_params == 1:
                    _args = [datum]
                else:
                    _args = [datum, maybe_other_datum]
                metrics.update(metric(*_args))
            return metrics

