from torch.utils.data import DataLoader

from .batch import MoleculeCollator
from .dataset import MoleculeDataset


class MoleculeDataLoader(DataLoader):
    """
    PyTorch DataLoader for Molecule Datasets

    Arguments:
    ----------
    dataset : MoleculeDataset
        the dataset to be loaded
    collate_fn : MoleculeCollator
        the method by which list of MoleculeDatum items are collated
    """

    def __init__(self, dataset: MoleculeDataset, collator: MoleculeCollator, **kwargs):
        super().__init__(dataset, collate_fn=collator.collate, **kwargs)
