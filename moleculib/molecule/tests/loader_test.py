from moleculib.molecule import batch, dataset, utils
from torch.utils.data import DataLoader

def test_dataloader():
    pids = ["1BFV", "2GN4", "5SE2", "5SE3"]
    bs = 2
    check_dataloader_batch_size(pids, bs)


def test_dataload_from_filesystem():
    bs = 3
    data_path = "/u/ilanm/moleculib/moleculib/data/pids_sanity.txt"
    pids = utils.pids_file_to_list(data_path)
    check_dataloader_batch_size(pids, bs)


def check_dataloader_batch_size(pids, bs):
    ds = dataset.MoleculeDataset.build(pids, max_workers=3)
    dataloader = DataLoader(
        ds, collate_fn=batch.MoleculePadBatch.collate, batch_size=bs#, drop_last=True
    )
    for batch_ in iter(dataloader):
        # assert batch_.atom_token.shape[0] == bs
        print(batch_.to_dict())

test_dataload_from_filesystem()