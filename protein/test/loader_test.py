import unittest

from protein.dataset import ProteinDataset
from protein.loader import ProteinDataLoader
from protein.batch import PadBatch
from protein.utils import pids_file_to_list


class DataLoaderTest(unittest.TestCase):

    def test_dataloader(self):
        pids = ["1BFV", "2GN4", "5SE2", "5SE3"]
        bs = 2
        self.check_dataloader_batch_size(pids, bs)

    def test_dataload_from_filesystem(self):
        bs = 2
        data_path = "data/pids_sanity.txt"
        pids = pids_file_to_list(data_path)
        self.check_dataloader_batch_size(pids, bs)

    def check_dataloader_batch_size(self, pids, bs):
        dataset = ProteinDataset.fetch_from_pdb(pids)
        dataloader = ProteinDataLoader(dataset, collator=PadBatch, batch_size=bs)
        for batch in dataloader:
            self.assertTrue(batch.atom_token.shape[0] == bs)
