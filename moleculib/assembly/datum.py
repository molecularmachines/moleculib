
from moleculib.molecule.datum import MoleculeDatum
from moleculib.protein.datum import ProteinDatum
from typing import List
from biotite.database import rcsb
import biotite.structure.io.pdb as pdb
from biotite.structure import filter_amino_acids


class AssemblyDatum:

    def __init__(
            self, 
            idcode: str = None,
            protein_data: List[ProteinDatum] = None,
            molecule_data: List[MoleculeDatum] = None,
        ):
        self.idcode = idcode
        self.protein_data = protein_data
        self.molecule_data =  molecule_data
    
    def from_datalist(datalist):
        protein_data = [datum for datum in datalist if datum.is_protein()]
        return AssemblyDatum(protein_data=protein_data)

    def filter_proteins(self, keep=None, drop=None):
        """ indices to keep or drop """
        if keep is not None:
            new_protein_data = [self.protein_data[i] for i in keep]
        elif drop is not None:
            new_protein_data = [self.protein_data[i] for i in range(len(self.protein_data)) if i not in drop]
        else: 
            raise ValueError('Either keep or drop must be provided')
        return AssemblyDatum(protein_data=new_protein_data)

    def plot(self, view, viewer=None, protein_style=dict(), molecule_style=dict()):
        if self.protein_data:
            for protein in self.protein_data:
                view = protein.plot(view, viewer, **protein_style)
        if self.molecule_data:
            for molecule in self.molecule_data:
                view = molecule.plot(view, viewer, **molecule_style)
        return view
    
    @classmethod 
    def from_filepath(
        cls, 
        filepath, 
        idcode=None,
    ):
        pdb_file = pdb.PDBFile.read(filepath)
        atom_array = pdb.get_structure(pdb_file, model=1)
        if idcode is None:
            idcode = str(filepath).split("/")[-1].split(".")[0]

        aa_filter = filter_amino_acids(atom_array)
        atom_array = atom_array[aa_filter]

        chains = set(atom_array.chain_id.tolist())

        proteins = []
        for chain in chains:
            chain_atom_array = atom_array[(atom_array.chain_id == chain)]
            if len(chain_atom_array) == 0: continue
            header = dict(
                idcode=idcode + chain,
                resolution=None,
            )
            protein = ProteinDatum.from_atom_array(chain_atom_array, header=header)
            proteins.append(protein)

        return AssemblyDatum(
            idcode=idcode,
            protein_data=proteins
        )

    @classmethod
    def fetch_pdb_id(
        cls, 
        idcode, 
        save_path=None
    ):
        filepath = rcsb.fetch(idcode, 'pdb', save_path)
        return cls.from_filepath(
            filepath, 
            idcode=idcode
        )
    