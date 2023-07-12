import os

class MoleculeDatum:
    """
    Incorporates molecular data to MoleculeDatum
    """
    def __init__(self, idcode, atom_token, atom_coord):
        self.idcode=idcode
        self.atom_token=atom_token
        self.atom_coord=atom_coord
        
    @classmethod
    def from_filename(cls, filename):
        with open(f'archive/{filename}', 'r') as f:
        atom_token=[]
        atom_coord=[]
        atom_charge=[]
        data=f.read().split('\n')
        n=int(data[0])
        for i in range(n):
            atom=data[2+i].split('\t')
            atom_token.append(atom[0])
            atom_coord.append(atom[1:4])
            atom_charge.append(atom[4])
        return()
        
    @classmethod
    def from_id(cls, n):
        nam='dsgdb9nsd_'+str(n)+'.xyz'
        return(cls.from_filename(nam))
