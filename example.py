import moleculib as mol
import torch

if __name__ == "__main__":
    # fetch all pdbs as Protein instances
    all_proteins = mol.fetch_proteins_from_pdb("all")

    # split dataset to reproducible train/val/test
    train, test, val = mol.split_dataset(all_proteins)

    # construct dataloaders
    train_dataloader = mol.ProteinDataLoader(
        train, batch_size=8, collate_fn="pad", attrs=["sequence", "coords"]
    )

    test_dataloader = mol.ProteinDataLoader(
        test, batch_size=8, collate_fn="pad", attrs=["sequence", "coords"]
    )
    # arbitrary torch neural net
    model = torch.nn.Module()
    optimizer = torch.optim.AdamW(model.params(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train network
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = batch.to(device)
            seqs, coords = batch
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
# checking i can write
