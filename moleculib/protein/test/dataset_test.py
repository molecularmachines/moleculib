import unittest

from .. import dataset


class DatasetTest(unittest.TestCase):

    def test_dataset(self):
        pdb_ids = ["1BFV", "2GN4"]
        ds = dataset.ProteinDataset.build(pdb_ids=pdb_ids)
        # test files have been fetched to file system
        self.assertTrue(len(ds) == len(pdb_ids))
