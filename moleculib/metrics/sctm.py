import os
from subprocess import Popen, PIPE, STDOUT

import pandas as pd
from colabdesign.mpnn import mk_mpnn_model

mpnn_model = mk_mpnn_model()


def get_designability_metrics(pdb_filepath, mpnn_model, num_seqs=8, temp=0.1):
    mpnn_model.prep_inputs(pdb_filename=pdb_filepath)
    samples = mpnn_model.sample(temperature=0.1, batch=num_seqs)
    sequences = samples['seq']
    return sequences


def predict_omegafold_metrics(sequences, output_directory,python_env="ophiuchusenv"):
    dir_path = os.getcwd()
    input_file = "input_file.fasta"
    input_filepath = os.path.join(dir_path, input_file)
    with open(input_filepath, "w") as f:
        for idx, seq in enumerate(sequences):
            f.write(">seq{}\n".format(idx))
            f.write("{}\n".format(seq))

    command_to_run = "conda activate {};omegafold {} {}".format(python_env,input_filepath, output_directory)
    process = Popen(command_to_run, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
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
