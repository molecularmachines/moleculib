import os
import pickle
import traceback
from functools import partial
from pathlib import Path
from tempfile import mkdtemp
from typing import List, Union
import requests

import random
import biotite
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map

from alphabet import UNK_TOKEN
from datum import NucleicDatum, dna_res_tokens, rna_res_tokens
from utils import pids_file_to_list
from tqdm import tqdm

#NOTE: TBD:
MAX_COMPLEX_SIZE = 32
PDB_HEADER_FIELDS = [
    ("idcode", str),
    ("num_res", int),
    ("num_rna_chains",int),
    ("num_dna_chains",int),
    ("standard", bool),
    ("resolution", float),
]
CHAIN_COUNTER_FIELDS = [(f"num_res_{idx}", int) for idx in range(MAX_COMPLEX_SIZE)]
PDB_METADATA_FIELDS = PDB_HEADER_FIELDS + CHAIN_COUNTER_FIELDS

SAMPLE_PDBS = ["5F9R", "2QK9", "8G8E", "8G8G", "8GME", "8H1T", "8H7A", "8HKC", "8HML"]



class PDBDataset(Dataset):
    """
    Holds NucleicDatum dataset with specified PDB IDs

    Arguments:
    ----------
    base_path : str
        directory to store all PDB files
    pdb_ids : List[str]
        list of all protein IDs that should be in the dataset
    format : str
        the file format for each PDB file, either "npz" or "pdb"
    attrs : Union[str, List[str]]
        a partial list of protein attributes that should be in each protein
    """

    def __init__(
        self,
        base_path: str,
        transform: None, #ProteinTransform = None,
        attrs: Union[List[str], str] = "all",
        metadata: pd.DataFrame = None,
        max_resolution: float = None,
        min_sequence_length: int = None,
        max_sequence_length: int = None,
        frac: float = 1.0,
        preload: bool = False,
        preload_num_workers: int = 10,
    ):
        super().__init__()
        self.base_path = Path(base_path)
        if metadata is None:
            with open(str(self.base_path / "metadata.pyd"), "rb") as file:
                metadata = pickle.load(file)
        self.metadata = metadata
        self.transform = transform

        if max_resolution is not None:
            self.metadata = self.metadata[self.metadata["resolution"] <= max_resolution]

        if min_sequence_length is not None:
            self.metadata = self.metadata[
                self.metadata["num_res"] >= min_sequence_length
            ]

        if max_sequence_length is not None:
            self.metadata = self.metadata[
                self.metadata["num_res"] <= max_sequence_length
            ]

        # shuffle and sample
        self.metadata = self.metadata.sample(frac=frac).reset_index(drop=True)

        # specific protein attributes
        nuc_attrs = [
            "idcode",
            "resolution",
            "sequence", #added
            "nuc_token", #mod
            "nuc_index",#mod
            "nuc_mask",#mod
            "chain_token",
            "atom_token",
            "atom_coord",
            "atom_mask",
        ]

        if attrs == "all":
            self.attrs = nuc_attrs
        else:
            for attr in attrs:
                if attr not in nuc_attrs:
                    raise AttributeError(f"attribute {attr} is invalid")
            self.attrs = attrs

    def _is_in_filter(self, sample):
        return int(sample["id"]) in self.shard_indices

    def __len__(self):
        return len(self.metadata)

    def load_index(self, idx):
        header = self.metadata.iloc[idx]
        pdb_id = header["idcode"]
        filepath = os.path.join(self.base_path, f"{pdb_id}.mmtf")
        molecules = NucleicDatum.from_filepath(filepath)
        return self.parse(header, molecules)

    def parse(self, molecules):
        raise NotImplementedError("PDBDataset is an abstract class")

    def __getitem__(self, idx):
        molecule = self.data[idx] if hasattr(self, "data") else self.load_index(idx)
        if self.transform is not None:
            for transformation in self.transform:
                molecule = transformation.transform(molecule)
        return molecule

    @staticmethod
    def _extract_statistics(datum):
        """
        Extracts statistics for a given datum.
        This function is called by `_maybe_fetch_and_extract` to retrieve the following for the datum:
            - idcode#
            - num_res#
            - num_rna_chains#
            - num_dna_chains#
            - standard#
            - resolution#
        Parameters:
            datum: NucleicDatum inst
        Returns:
            df: A df containing the extracted statistics for the datum.
        """
        is_standard = not (datum.nuc_token == UNK_TOKEN).all()
        metrics = dict(
            idcode=datum.idcode,
            standard=is_standard,
            num_rna_chains=0,
            num_dna_chains=0,
            resolution=datum.resolution,
            num_res=len(datum.sequence),
        )

        def check_strictly_increasing(lst):
            for i in range(1, len(lst)):
                if lst[i] < lst[i-1]:
                    return False
            return True



        for chain in range(np.max(datum.chain_token) + 1):
            #getting chain length for each chain
            chain_residues = (datum.chain_token == chain) #bool list, true where the chain is
            num_residues = chain_residues.sum()
            metrics[f"num_res_{chain}"] = num_residues

            #NOTE: this check is prob not relevant, tbd if needed
            if check_strictly_increasing(datum.chain_token) == False:
                raise Exception("The datum chain tokens are not strictly increasing")
            #getting chain type
            chain_indices = [i for i, val in enumerate(chain_residues) if val]
            random_index = random.choice(chain_indices)
            random_nuc_token = datum.nuc_token[random_index]
            if random_nuc_token in rna_res_tokens:
                metrics['num_rna_chains']+=1
            elif random_nuc_token in dna_res_tokens:
                metrics['num_dna_chains']+=1
            # else:
            #     print(random_nuc_token)
            #     raise Exception("The datum nuc token didn't fit RNA or DNA tokens")
        return Series(metrics).to_frame().T

    @staticmethod
    def _maybe_fetch_and_extract(pdb_id, save_path):
        """
        This function is called by the `build` function to check if the datum for the given PDB ID can be fetched. 
        If it can be fetched, the function retrieves the datum and extracts its statistics.

        Parameters:
            pdb_id (str): The PDB ID of the datum to fetch and extract.
            save_path (str): The path to save the fetched datum.

        Returns:
            tuple or None: A tuple containing the instance of the fetched datum and its extracted statistics 
                if the fetch is successful, otherwise None and error usually.
        """
        try:
            datum = NucleicDatum.fetch_pdb_id(pdb_id, save_path=save_path)
        except KeyboardInterrupt:
            exit()
        except (ValueError, IndexError) as error:
            print(traceback.format_exc())
            print(error)
            return None
        except biotite.database.RequestError as request_error:
            print(request_error)
            return None
        if len(datum.sequence) == 0:
            return None
        return (datum, PDBDataset._extract_statistics(datum))


    @classmethod
    def build(
        cls,
        pdb_ids: List[str] = None,
        save: bool = True,
        save_path: str = None,
        max_workers: int = 20,
        **kwargs,
    ):
        """
        Builds dataset from scratch given specified pdb_ids, prepares
        data and metadata for later use.
        """
        print(f"Extracting {len(pdb_ids)} PDB IDs with {max_workers} workers...")
        if pdb_ids is None:
            root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
            pdb_ids = pids_file_to_list(root + "/data/pids_all.txt")
        if save_path is None:
            save_path = mkdtemp()

        series = {c: Series(dtype=t) for (c, t) in PDB_METADATA_FIELDS}
        metadata = DataFrame(series)
        
        extractor = partial(cls._maybe_fetch_and_extract, save_path=save_path)
        if max_workers > 1:
            extraction = process_map(
                extractor, pdb_ids, max_workers=max_workers, chunksize=50
            )
        else:
            extraction = list(map(extractor, pdb_ids))

        extraction = filter(lambda x: x, extraction)
        
        data, metadata_ = list(map(list, zip(*extraction)))
        metadata = pd.concat((metadata, *metadata_), axis=0)

        if save:
            with open(str(Path(save_path) / "metadata.pyd"), "wb") as file:
                pickle.dump(metadata, file)

        return cls(base_path=save_path, metadata=metadata,transform=None, **kwargs)

def get_pdb_ids():
    url = 'https://data.rcsb.org/rest/v1/core/entry/'
    params = {
        'entity_poly.rcsb_entity_polymer_type': 'DNA',  # Filter by DNA molecules
        'rcsb_entry_info.resolution_combined.operator': '<',  # Filter by experimental method (X-ray)
        'limit': 100,  # Adjust this value to fetch more or fewer entries per request
    }


    response = requests.get(url, params=params)
    data = response.json()
    breakpoint()
    pdb_ids = [entry['rcsb_id'] for entry in data['result_set']]
    return pdb_ids

if __name__ == '__main__':

    # print("hello")
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # with open('/u/danaru/moleculib/moleculib/moleculib/data/pids_all.txt', 'r') as file:
    #     data = file.read()
    #     pdbs = data.split(',')
    
    
    # dataset = PDBDataset.build(pdbs, save_path = script_dir)
    # breakpoint()
    ###Check it saves to where i want it to save:
    # print("hey")
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # # Create a dummy file
    # with open("/u/danaru/moleculib/moleculib/moleculib/nucleic/dummy_file.txt", "w") as f:
    #     f.write("This is a dummy file.")
    #     print("saved")
    # print(script_dir)


    #check permissions:
    # directory_path = "/u/danaru/moleculib/moleculib/moleculib/nucleic"

    # # Get the permissions of the directory
    # permissions = os.stat(directory_path).st_mode

    # # Check if the directory is writable
    # is_writable = bool(permissions & 0o200)

    # if is_writable:
    #     print(f"The directory '{directory_path}' is writable.")
    # else:
    #     print(f"The directory '{directory_path}' is not writable.")
    print("new")
    with open('/u/danaru/moleculib/moleculib/moleculib/data/pids_all.txt', 'r') as file:
        data = file.read()
        pdbs = data.split(',')
    print(len(pdbs))
    
    
    # dataset = PDBDataset.build(pdbs)
    # breakpoint()
    # # Call the function to get a list of PDB IDs
    # all_pdb_ids = get_pdb_ids()
    # print(all_pdb_ids)

    # print(d.metadata)  
    # breakpoint()


    # # Call the function to get a list of PDB IDs
    # all_pdb_ids = get_pdb_ids()
    # print(all_pdb_ids)

    # print(d.metadata)  
    # breakpoint()