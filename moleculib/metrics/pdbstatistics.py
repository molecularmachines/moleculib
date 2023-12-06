# biotite dependencies
import sys
from subprocess import Popen, PIPE, STDOUT

sys.path.append("/u/manvitha/PyRosetta4.Debug.python310.linux.release-364/")
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBParser
import pyrosetta;

pyrosetta.init()
from pyrosetta import *
from pyrosetta.teaching import *

init()

sec_struct_codes = {0: "I",
                    1: "S",
                    2: "H",
                    3: "E",
                    4: "G",
                    5: "B",
                    6: "T",
                    7: "C"}
dssp_to_abc = {"I": "c",
               "S": "c",
               "H": "a",
               "E": "b",
               "G": "c",
               "B": "b",
               "T": "c",
               "C": "c"}

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        if not "CRYST1" in content:
            f.seek(0, 0)
            f.write(line.rstrip('\r\n') + '\n' + content)
        f.close()
    return True


class PDBStatistics:
    """
    Incorporates protein data to MolecularDatum
    and reshapes atom arrays to residue-based representation
    """

    def __init__(
            self,
            idcode: str,
            pdb_path: str,
            **kwargs,
    ):
        self.pdb_path = pdb_path
        str_to_add = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1 "
        line_prepender(self.pdb_path, str_to_add)
        self.biopython_structure = PDBParser().get_structure(idcode, self.pdb_path)
        self.biopython_model = self.biopython_structure[0]
        seq = []
        for residue in self.biopython_model:
            #             seq.append(d3to1[residue.resname])
            seq.append("X")
        self.sequence = "".join(seq)

    def __len__(self):
        return len(self.sequence)

    def get_rosetta_statistics(self):
        pose = pyrosetta.pose_from_pdb(self.pdb_path)
        sfxn = get_score_function(True)
        score = sfxn(pose)
        return score

    def get_secondary_structure(self):
        dssp = DSSP(self.biopython_model, self.pdb_path, file_type='PDB')
        sec_structure = ''
        for z in range(len(dssp)):
            a_key = list(dssp.keys())[z]
            sec_structure += dssp[a_key][2]
        return sec_structure

    def get_term_motifs(self, python_env="ophiuchusenv",
                        run_path="/mas/projects/molecularmachines/binaries/MASTER/termanal/run.py"):
        command_to_run = "conda activate {};python {} --p {}".format(python_env, run_path, self.pdb_path)
        process = Popen(command_to_run, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        process.wait()
        return process.returncode






