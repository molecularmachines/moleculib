standard_aminoacids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

# read lines of components.cif:
with open('standard_components.cif', 'r') as f:
    lines = f.readlines()

# find starting points of each standard aminoacid:
standard_lines = [] 
for aminoacid in standard_aminoacids:
    for i, line in enumerate(lines):
        if line.startswith('data_{0}'.format(aminoacid)):
            i = i + 1
            curr_line = lines[i]
            standard_lines.append(f'{aminoacid}\natoms\n')
            while not curr_line.startswith('_chem_comp_atom.pdbx_ordinal'):
                i = i + 1
                curr_line = lines[i]
            while not curr_line.startswith('#   #'):
                i = i + 1
                curr_line = lines[i]
                standard_lines.append(curr_line)
            standard_lines.append('bonds\n')
            while not curr_line.startswith('_chem_comp_bond.pdbx_ordinal'):
                i = i + 1
                curr_line = lines[i]
            while not curr_line.startswith('#   #'):
                i = i + 1
                curr_line = lines[i]
                standard_lines.append(curr_line)
            print(aminoacid, i)
            break

with open('filtered_components.cif', 'w') as f:
     f.writelines(standard_lines)


# import and use biotite to read the standard aminoacids:
# import biotite.structure as struc
# import biotite.structure.io as strucio
# import biotite.structure.info as strucinfo
# import biotite.structure.info as strucinfo

# # read the standard aminoacids:
# standard_aminoacids = strucio.load_structure('standard_components.cif')

# import and use biopython to read the standard aminoacids:
# from Bio.PDB import MMCIFParser
# from Bio.PDB.Polypeptide import three_to_one

# read the standard aminoacids:
# parser = MMCIFParser()
# standard_aminoacids = parser.get_structure('standard_components', 'standard_components.cif')

# iterate over residues: