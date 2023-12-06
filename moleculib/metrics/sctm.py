import os,sys
from subprocess import Popen, PIPE, STDOUT

import biotite.structure.io as bsio
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBIO import PDBIO
from pathlib import Path

def get_designability_metrics(pdb_filepath, mpnn_model, num_seqs=8, temp=0.1):
    mpnn_model.prep_inputs(pdb_filename=pdb_filepath)
    samples = mpnn_model.sample(temperature=0.1, batch=num_seqs)
    sequences = samples['seq']
    return sequences


def predict_omegafold_metrics(idcode,sequences, output_directory, python_env="ophiuchusenv"):
    dir_path = os.getcwd()
    input_file = "input_file.fasta"
    input_filepath = os.path.join(dir_path, input_file)
    with open(input_filepath, "w") as f:
        for idx, seq in enumerate(sequences):
            f.write(">{}_seq{}\n".format(idcode,idx))
            f.write("{}\n".format(seq))

    command_to_run = "conda activate {};omegafold {} {}".format(python_env, input_filepath, output_directory)
    process = Popen(command_to_run, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(line.strip())
        sys.stdout.flush()
    
    process.wait()
    return process.returncode


def tmalign_metrics(tmalign_path, reference_pdb_filepath, folder_of_pdbs, tmalign_outputs_path):
    command_to_run = ""
    for filename in os.listdir(folder_of_pdbs):
        filepath = os.path.join(folder_of_pdbs, filename)
        logpath = os.path.join(tmalign_outputs_path, filename.replace(".pdb", ".log"))
        command_to_run = command_to_run + "{} {} {} > {}; ".format(tmalign_path, reference_pdb_filepath, filepath,
                                                                   logpath)
    process = Popen(command_to_run, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    process.wait()
    return process.returncode


def get_plddt_score(pdb_path):
    struct = bsio.load_structure(pdb_path, extra_fields=["b_factor"])
    bmean = struct.b_factor.mean()
    return bmean


def get_pdb_from_id(pdb_id, chain_id, save_to):
    pdbl = PDBList()
    pdb_filepath = os.path.join(save_to, "pdb"+pdb_id + ".ent")
    if Path(pdb_filepath).is_file():
        print("exists already")
    else:
        native_pdb = pdbl.retrieve_pdb_file(pdb_id, pdir=save_to, file_format='pdb')
        
    parser = PDBParser()
    structure = parser.get_structure(pdb_id, pdb_filepath)
    
    io = PDBIO()

    for chain in structure.get_chains(): 
        io.set_structure(chain)
        io.save(os.path.join(save_to, pdb_id + chain_id + ".pdb"))
   
    return os.path.join(save_to, pdb_id + chain_id + ".pdb")


def parse_tmalign_scores(tmalign_logs_path):
    tmrmsds = []
    tmscores = []
    filenames = []
    for filename in os.listdir(tmalign_logs_path):
        filepath = os.path.join(tmalign_logs_path, filename)
        filenames.append(filepath)
        with open(filepath, "r") as f:

            lines = f.readlines()
            tmscor = None
            for line in lines:
                if "RMSD=" in line:
                    tmrmsds.append(float(line.split()[4].replace(",", "")))
                elif "TM-score=" in line and tmscor is None:
                    tmscor = "Added"
                    tmscores.append(float(line.split()[1].replace(",", "")))

    return pd.DataFrame({
        "tmrmsds": tmrmsds,
        "tmscores": tmscores,
        "filenames": filenames
    })
