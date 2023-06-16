from collections import OrderedDict
from typing import List

import numpy as np
from ordered_set import OrderedSet

# from https://x3dna.org/articles/name-of-base-atoms-in-pdb-formats#:~:text=Canonical%20bases%20(A%2C%20C%2C,%2C%20C8%2C%20N9)%20respectively.

#ensure order and keep track of what atoms are present or missing
base_atoms = OrderedDict(
        A=["N1","C2","N3", "C4", "C5", "C6", "N7", "C8", "N9"], #calman had N6
        U=["N1", "C2", "O2", "N3", "C4", "C5", "C6"], #
        T=["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C5M", "C6"], #Calman Didn't have C5M but had:
        #T=['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6']
        G=["N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"], #same
        C=["N1", "C2", "O2", "N3", "C4", "C5", "C6"] #Calman had extra N4
)

