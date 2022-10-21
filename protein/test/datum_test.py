import os
from pathlib import Path
import unittest
from tqdm import tqdm
from tempfile import gettempdir

from protein.datum import ProteinDatum
from protein.utils import pids_file_to_list


class ProteinDatumTest(unittest.TestCase):
    def check_attribute_shapes(self, sample):
        self.assertTrue(sample.atom_coord.shape[1:] == (14, 3))
        self.assertTrue(sample.atom_token.shape[1:] == (14,))
        self.assertTrue(sample.atom_mask.shape[1:] == (14,))
        self.assertTrue(len(sample.residue_token.shape) == 1)
        self.assertTrue(len(sample.residue_mask.shape) == 1)

    def check_masking(self, sample):
        self.assertFalse((sample.atom_coord[sample.atom_mask] == 0.0).sum())
        self.assertTrue((sample.atom_coord[~sample.atom_mask] == 0.0).sum())

    def test_from_pdb(self):
        sample = ProteinDatum.fetch_from_pdb("1BFV")
        self.check_attribute_shapes(sample)
        self.check_masking(sample)

    def test_from_npz(self):
        sample = ProteinDatum.fetch_from_pdb(
            "1BFV", save_path=gettempdir(), format="npz"
        )
        npz_path = Path(gettempdir()) / "1BFV.npz"
        sample = ProteinDatum.from_filepath(npz_path, format='npz')
        self.check_attribute_shapes(sample)
        self.check_masking(sample)

        # clear temp file
        os.remove(str(npz_path))

    def test_all_pdb(self):
        all_pids_path = "data/pids_all.txt"
        pids = pids_file_to_list(all_pids_path)
        for pid in pids:
            protein = ProteinDatum.fetch_from_pdb(pid)
