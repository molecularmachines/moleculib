import unittest

from protein.dataset import ProteinDataset


class DatasetTest(unittest.TestCase):

    def test_dataset(self):
        pdb_ids = ["1BFV", "2GN4"]
        dataset = ProteinDataset.fetch_from_pdb(pdb_ids=pdb_ids)

        # test files have been fetched to file system
        self.assertTrue(len(dataset) == len(pdb_ids))
