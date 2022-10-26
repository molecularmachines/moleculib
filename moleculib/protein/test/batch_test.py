import unittest
import numpy as np

from .. import datum
from .. import batch


class BatchTest(unittest.TestCase):
    def check_pad_batch_attribute_shapes(self, sample, batch_size, max_seq_size):
        self.assertTrue(sample.atom_coord.shape == (batch_size, max_seq_size, 14, 3))
        self.assertTrue(sample.atom_token.shape == (batch_size, max_seq_size, 14))
        self.assertTrue(sample.atom_mask.shape == (batch_size, max_seq_size, 14))
        self.assertTrue(sample.residue_token.shape == (batch_size, max_seq_size))
        self.assertTrue(sample.residue_token.shape == (batch_size, max_seq_size))

    def check_unbatched(self, samples, samples_after):
        for (sample, sample_after) in zip(samples, samples_after):
            for attr, value in vars(sample).items():
                is_same_value = value == getattr(sample_after, attr)
                if type(value) is np.ndarray:
                    is_same_value = is_same_value.all()
                self.assertTrue(is_same_value)

    def check_masking(self, sample):
        self.assertFalse((sample.atom_coord[sample.atom_mask] == 0.0).sum())
        self.assertTrue((sample.atom_coord[~sample.atom_mask] == 0.0).sum())

    def test_pad_batch(self):
        samples = list(map(datum.ProteinDatum.fetch_from_pdb, ["1BFV", "2GN4"]))
        max_seq_size = max([len(sample.sequence) for sample in samples])
        batch_ = batch.PadBatch.collate(samples)
        self.check_pad_batch_attribute_shapes(
            batch_, batch_size=2, max_seq_size=max_seq_size
        )
        self.check_masking(batch_)

    def test_pad_unbatch(self):
        samples = list(map(datum.ProteinDatum.fetch_from_pdb, ["1BFV", "2GN4"]))
        max_seq_size = max([len(sample.sequence) for sample in samples])
        batched_samples = batch.PadBatch.collate(samples)
        samples_after = batched_samples.revert()
        self.check_unbatched(samples, samples_after)

    def check_geometric_batch_attribute_shapes(self, sample, total_num_nodes):
        self.assertTrue(sample.atom_coord.shape == (total_num_nodes, 14, 3))
        self.assertTrue(sample.atom_token.shape == (total_num_nodes, 14))
        self.assertTrue(sample.atom_mask.shape == (total_num_nodes, 14))
        self.assertTrue(sample.residue_token.shape == (total_num_nodes,))
        self.assertTrue(sample.residue_token.shape == (total_num_nodes,))

    def test_geometric_batch(self):
        samples = list(map(datum.ProteinDatum.fetch_from_pdb, ["1BFV", "2GN4"]))
        num_nodes = [len(sample.sequence) for sample in samples]
        total_num_nodes = sum(num_nodes)
        batch_ = batch.GeometricBatch.collate(samples)
        _, counts = np.unique(batch_.batch_index, return_counts=True)
        self.assertTrue(num_nodes == counts.tolist())
        self.check_geometric_batch_attribute_shapes(batch_, total_num_nodes)

    def test_geometric_unbatch(self):
        samples = list(map(datum.ProteinDatum.fetch_from_pdb, ["1BFV", "2GN4"]))
        batched_samples = batch.GeometricBatch.collate(samples)
        samples_after = batched_samples.revert()
        self.check_unbatched(samples, samples_after)
