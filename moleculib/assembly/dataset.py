import os
import pickle
from typing import Callable, List

from tqdm import tqdm

from ..abstract.dataset import PreProcessedDataset


class ChromaDataset(PreProcessedDataset):

    def __init__(
        self,
        base_path,
        transform: List[Callable] = None,
        shuffle=True,
        reduced=False,
    ):
        name = "CHROMA_FULL" if not reduced else "CHROMA_TINY"
        base_path = os.path.join(base_path, f"{name}.pyd")

        with open(base_path, "rb") as fin:
            print("Loading data...")
            splits = pickle.load(fin)

        super().__init__(splits, transform, shuffle, pre_transform=False)
