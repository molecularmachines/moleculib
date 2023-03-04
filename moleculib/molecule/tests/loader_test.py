from moleculib.molecule import batch, dataset, loader, utils


def test_dataloader():
    pids = ["1BFV", "2GN4", "5SE2", "5SE3"]
    bs = 2
    check_dataloader_batch_size(pids, bs)


def test_dataload_from_filesystem():
    bs = 2
    data_path = "moleculib/moleculib/data/pids_sanity.txt"
    pids = utils.pids_file_to_list(data_path)
    check_dataloader_batch_size(pids, bs)


def check_dataloader_batch_size(pids, bs):
    ds = dataset.MoleculeDataset.build(pids, max_workers=3)
    dataloader = loader.MoleculeDataLoader(
        ds, collator=batch.MoleculePadBatch, batch_size=bs, drop_last=True
    )
    for batch_ in dataloader:
        assert batch_.atom_token.shape[0] == bs
