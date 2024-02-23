from scipy.linalg.decomp import nonzero
# Standard libraries
# Python Packages
import pandas as pd
import numpy as np
# Deep Learning ones
import torch
# from torch.utils.data import Dataset, DataLoader
# from Bio.PDB import PDBParser
# from Bio.SeqUtils import seq1
# from transformers import T5Tokenizer, T5EncoderModel
from typing import List, Tuple
from biotite.sequence.io import fasta

alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                         'Y', "X"]

def onehot_encode_datum(sequencedatum:SeqDatum):
    residue_tokens = sequencedatum.residue_token
    one_hot_tensor = torch.zeros(len(residue_tokens), len(all_residues))
    for i, token in enumerate(residue_tokens):
        one_hot_tensor[i, token] = 1
    return one_hot_tensor

def decode_one_hot(x):
    decoded = np.argmax(x, axis=2)
    return decoded

def residue_tokens_to_sequence(sequencedatum:SeqDatum):
    sequence_list = [all_residues[i] for i in sequencedatum.residue_token]
    residues=[]
    for name in sequence_list:
        if name!="MASK":
            residues.append(ProteinSequence.convert_letter_3to1(name))
        else:
            residues.append("U")
#     residues = [ProteinSequence.convert_letter_3to1(name) for name in sequence_list if name != "MASK"]
    return "".join(residues)

def load_fasta_file(input: str) -> List[Tuple[str]]:
    # read input fasta file
    fasta_file = fasta.FastaFile.read(input)
    fasta_sequences = fasta.get_sequences(fasta_file)
    sequences = list(fasta_sequences.values())
    sequences = [str(s) for s in sequences]
    names = list(fasta_sequences.keys())
    all_sequences = {}
    for k,v in zip(names,sequences):
        all_sequences[k]=v
    return all_sequences

def save_fasta_file(sequences, names, save_path):
    step_fasta_file = fasta.FastaFile()
    for j, res_name in enumerate(names):
        step_fasta_file[res_name] = sequences[j]
    step_fasta_file.write(save_path)
    return save_path
    
def one_hot_encode_seq(protein_seq, max_length=14):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    encoded_protein = np.zeros((max_length, len(alphabet)))

    integer_encoded = [char_to_int[char] for char in protein_seq]
    one_hot_array = []
    for idx, i in enumerate(integer_encoded):
        letter = [0 for _ in range(len(alphabet))]
        letter[i] = 1
        one_hot_array.append(letter)
    pad_length = max_length - len(protein_seq)
    pad_hot_array = []
    for i in range(0, pad_length):
        letter = [0 for _ in range(len(alphabet))]
        pad_hot_array.append(letter)
    return np.array(one_hot_array + pad_hot_array)

# def encode_seq_with_ankh(protein_seq):
#     model, tokenizer = ankh.load_large_model()
#     model.eval()
#     embeddings = None
#     protein_sequences = [list(seq) for seq in protein_seq]
#     outputs = tokenizer.batch_encode_plus(protein_sequences,
#                                                add_special_tokens=True,
#                                                padding=True,
#                                                is_split_into_words=True,
#                                                return_tensors="pt")
#     with torch.no_grad():
#         embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])

#     return embeddings.last_hidden_state

# def encode_seq_with_protT5(protein_seq):
#     tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
#     model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
#     embeddings = None
#     seqs = [list(seq) for seq in protein_seq]
#     protein_sequences = []
#     for sequence in seqs:
#         sequence = "".join(list(sequence))
#         protein_sequences.append(" ".join(list(re.sub(r"[UZOB]", "X", sequence))))

#     # tokenize sequences and pad up to the longest sequence in the batch
#     ids = tokenizer.batch_encode_plus(protein_sequences, add_special_tokens=True, padding="longest")

#     input_ids = torch.tensor(ids['input_ids']).to(device)
#     attention_mask = torch.tensor(ids['attention_mask']).to(device)

#     # generate embeddings
#     with torch.no_grad():
#         embedding_rpr = model(input_ids=input_ids, attention_mask=attention_mask)

#     return embedding_rpr

# def encode_seq_with_esm(protein_seq):
#     model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#     batch_converter = esm_alphabet.get_batch_converter()
#     model.eval()  # disables dropout for deterministic results
#     embeddings = None

#     # Prepare raw_data
#     data = []
#     for idx, ps in enumerate(protein_seq):
#         data.append(('protein{}'.format(idx), ps))

#     batch_labels, batch_strs, batch_tokens = batch_converter(data)
#     batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)

#     # Extract per-residue representations (on CPU)
#     with torch.no_grad():
#         results = model(batch_tokens, repr_layers=[33], return_contacts=True)
#     token_representations = results["representations"][33]

#     sequence_representations = []
#     for i, tokens_len in enumerate(batch_lens):
#         sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
#     # num_seqsx1280
#     return sequence_representations[0]