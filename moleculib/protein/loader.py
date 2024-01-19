# from torch.utils.data import DataLoader

from .batch import ProteinCollator
from .dataset import ProteinDataset


class ProteinDataLoader:
    """
    PyTorch DataLoader for Protein Datasets

    Arguments:
    ----------
    dataset : ProteinDataset
        the dataset to be loaded
    collate_fn : ProteinCollator
        the method by which list of ProteinDatum items are collated
    """

    def __init__(self, dataset: ProteinDataset, collator: ProteinCollator, **kwargs):
        super().__init__(dataset, collate_fn=collator.collate, **kwargs)
