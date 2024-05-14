#biotite dependencies
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb
import biotite.database.uniprot as uniprot
from biotite.structure import get_residues
from biotite.sequence import ProteinSequence
import biotite.sequence.io.fasta as fasta
import numpy as np
from collections import OrderedDict
import biotite.database.entrez as entrez

from urllib.parse import urlparse, parse_qs, urlencode
import requests
from requests.adapters import HTTPAdapter, Retry

from moleculib.protein.alphabet import all_residues,get_residue_index

API_URL = "https://rest.uniprot.org"
allowed_aa= ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']


class SeqDatum:
    """
    Incorporates protein data to MolecularDatum
    and reshapes atom arrays to residue-based representation
    """

    def __init__(
        self,
        idcode: str,
        sequence: ProteinSequence,
        residue_token:np.ndarray,
        **kwargs,
    ):
        self.idcode = idcode
        self.sequence =sequence
        self.residue_token = residue_token
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        return len(self.sequence)
    
    def onehot_encode_datum(self):
        residue_tokens = self.residue_token
        one_hot_tensor = torch.zeros(len(residue_tokens), len(all_residues))
        for i, token in enumerate(residue_tokens):
            one_hot_tensor[i, token] = 1
        return one_hot_tensor

    @classmethod
    def empty_seqrecord(cls):
        return cls(
            idcode="",
            sequence=ProteinSequence(""),
            residue_token = None
        )

    @classmethod
    def from_tokens(cls, residues):
        """
        Fasta file would only have sequence entry
        :param filepath:
        :return:
        """
        sequence_list = []
        sequence_mask = []
        for name in residues:
            residue_name=all_residues[name]
            if residue_name!="PAD" and residue_name!="MASK":
                sequence_list.append(ProteinSequence.convert_letter_3to1(residue_name))
                sequence_mask.append(True)
            else:
                sequence_mask.append(False)
                
        return cls(
            idcode="",
            sequence="".join(sequence_list),
            residue_token=np.array(residues),
            sequence_mask =np.array(sequence_mask)
        )

    @classmethod
    def from_filepath(cls,filepath):
        """
        Fasta file would only have sequence entry
        :param filepath:
        :return:
        """
        fasta_file = fasta.FastaFile.read(filepath)
        sequence = [ProteinSequence(s) for s in fasta_file.values()][0]
        
        residues = [
            ("UNK" if (ProteinSequence.convert_letter_1to3(name) not in all_residues) else ProteinSequence.convert_letter_1to3(name)) for name in list(selected_sequence)
        ]
        
        residue_tokens = np.array(list(map(lambda res: get_residue_index(res), residues)))
      
        sequence_mask = []
        for name in residue_tokens:
            residue_name=all_residues[name]
            if residue_name!="PAD" and residue_name!="MASK":
                sequence_mask.append(True)
            else:
                sequence_mask.append(False)
                
        return cls(
            idcode="",
            sequence=sequence,
            residue_token=residue_tokens,
            sequence_mask=sequence_mask
        )
    
    @classmethod
    def from_sequence(cls,idcode,sequence):
        """
        Fasta file would only have sequence entry
        :param filepath:
        :return:
        """
         
        residues = [
            ("UNK" if (ProteinSequence.convert_letter_1to3(name) not in all_residues) else ProteinSequence.convert_letter_1to3(name)) for name in list(sequence)
        ]
        
        residue_tokens = np.array(list(map(lambda res: get_residue_index(res), residues)))

        return cls(
            idcode=idcode,
            sequence=sequence,
            sequence_mask=np.ones_like(residue_tokens).astype(np.bool_),
            residue_token=residue_tokens
        )
    
    @classmethod
    def from_pdb_id(cls, pdb_id,chain_id=None):
        pdbx_file = pdbx.PDBxFile.read(rcsb.fetch(pdb_id, "mmcif"))
        structure = pdbx.get_structure(pdbx_file,model=1)
        chain_structures = {}
        chain_sequence = {}
        for chain_id in list(set(structure.get_annotation('chain_id'))):
            chain_structure = structure[(structure.chain_id == chain_id)]
            _, res_names = get_residues(chain_structure)
            res_names = [name for name in res_names if name in allowed_aa]
            sequence = ProteinSequence(list(res_names))
            chain_sequence[chain_id]=sequence
            chain_structures[chain_id]=chain_structure
        selected_sequence = chain_sequence[chain_id]
        chain_id = chain_id
        selected_structure = chain_structures[chain_id]
        
        residues = [
            ("UNK" if (ProteinSequence.convert_letter_1to3(name) not in all_residues) else ProteinSequence.convert_letter_1to3(name)) for name in list(selected_sequence)
        ]
        
        residue_tokens = np.array(list(map(lambda res: get_residue_index(res), residues)))

        return cls(
            idcode=pdb_id,
            sequence=selected_sequence,
            residue_token=residue_tokens,
            structure=selected_structure,
            chain_id = chain_id
        )
    
    @classmethod
    def from_uniprot_id(cls, uniprotid,save_path=None):
        def map_uniprot_id(to_db, search_id):
            request = requests.post(
                f"{API_URL}/idmapping/run",
                data={"from": "UniProtKB_AC-ID", "to": to_db, "ids": search_id},
            )
            return request.json()["jobId"]
    
        filepath = uniprot.fetch(uniprotid, "fasta")
        fasta_file = fasta.FastaFile.read(filepath)

        #take only the first one
        sequence = [ProteinSequence(s) for s in fasta_file.values()][0]
        chain_id = ["chain_{}".format(i) for i in range(0,len(sequences))][0]
        query = uniprot.SimpleQuery("accession",uniprotid)

       
        residues = [
            ("UNK" if (ProteinSequence.convert_letter_1to3(name) not in all_residues) else ProteinSequence.convert_letter_1to3(name)) for name in list(sequence)
        ]
        residue_tokens = np.array(list(map(lambda res: get_residue_index(res), residues)))
        
        return cls(
            idcode=uniprotid,
            sequences=sequence,
            residue_token=residue_tokens,
            structure=None,
            chain_ids = chain_id
        )