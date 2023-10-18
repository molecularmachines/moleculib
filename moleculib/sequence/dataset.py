import json
import sys

sys.path.append("/u/manvitha/prose")

import torch
from embed.seq_embeddings import ProteinSeqEmbeddings
from torch.utils.data import Dataset


class BindingAffinityDataset(Dataset):
    def __init__(self, data, precomputed_embeddings_path, max_peptide_length=14,
                 max_receptor_length=373,
                 receptor_transform='esm',
                 peptide_transform='onehot'):
        self.data = data
        self.max_peptide_length = max_peptide_length
        self.max_receptor_length = max_receptor_length
        self.receptor_transform = receptor_transform
        self.peptide_transform = peptide_transform
        self.precomputed_embeddings_path = precomputed_embeddings_path
        self.precomputed_embeddings = None
        self.peptide_encoder = ProteinSeqEmbeddings(encoder_type='onehot', max_length=1400,
                                                                             token_level=False, device_name='cpu')
        with open(self.precomputed_embeddings_path, "r") as f:
            self.precomputed_embeddings = json.load(f)

    def __len__(self):
        """
        Total length of the dataset
        """
        return len(self.data)

    def _encode_receptor(self, allele):
        """
        Encode the receptor sequence - Recepter Sequence is identified by the allele type
        receptor_sequence - Allele Key
        """
        return torch.Tensor(self.precomputed_embeddings[allele]['embedding'])

    def _encode_peptide(self, peptide_sequence):
        encoded_peptide = torch.Tensor(self.peptide_encoder.one_hot_encode_seq(peptide_sequence))
        return encoded_peptide

    def __getitem__(self, idx):
        row = self.data.iloc[[idx]]
        allele = row["Allele"].values[0]
        peptide = row["ElutedLigand"].values[0]
        score = row["Score"].values[0]
        encoded_peptide = self._encode_peptide(peptide)
        encoded_receptor = self._encode_receptor(allele)
        return (encoded_receptor, encoded_peptide, torch.Tensor([score]))


class ElutedLigandDataset(Dataset):
    def __init__(self, data, precomputed_embeddings_path, max_peptide_length=14,
                 max_receptor_length=373,
                 receptor_transform='esm',
                 peptide_transform='onehot'):
        self.data = data
        self.max_peptide_length = max_peptide_length
        self.max_receptor_length = max_receptor_length
        self.receptor_transform = receptor_transform
        self.peptide_transform = peptide_transform
        self.precomputed_embeddings_path = precomputed_embeddings_path
        self.precomputed_embeddings = None
        self.peptide_encoder = protein_seq_embeddings = ProteinSeqEmbeddings(encoder_type='onehot', max_length=1400,
                                                                             token_level=False, device_name='cpu')
        with open(self.precomputed_embeddings_path, "r") as f:
            self.precomputed_embeddings = json.load(f)

    def __len__(self):
        """
        Total length of the dataset
        """
        return len(self.data)

    def _encode_receptor(self, allele):
        """
        Encode the receptor sequence - Recepter Sequence is identified by the allele type
        receptor_sequence - Allele Key
        """
        return torch.Tensor(self.precomputed_embeddings[allele]['embedding'])

    def _encode_peptide(self, peptide_sequence):
        encoded_peptide = torch.Tensor(self.peptide_encoder.one_hot_encode_seq(peptide_sequence))
        return encoded_peptide

    def __getitem__(self, idx):
        row = self.data.iloc[[idx]]
        allele = row["Allele"].values[0]
        peptide = row["ElutedLigand"].values[0]
        score = row["Presence"].values[0]
        encoded_peptide = self._encode_peptide(peptide)
        encoded_receptor = self._encode_receptor(allele)
        return (encoded_receptor, encoded_peptide, torch.Tensor([score]))
