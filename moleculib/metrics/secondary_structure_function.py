def sec_struc(pdb_file):
    ss_by_struc = {}
    structure_types = {'HELIX', 'SHEET'}
    
    def sec_struc_type(struc_type, filename):
        chains = []
        struc_dict = {}
        raw_line = []
        filename.seek(0)
        assert struc_type in {'HELIX', 'SHEET'}
        start, stop, chain_loc = 5,8,4
        if struc_type == 'SHEET':
            start, stop, chain_loc = 6,9,5
        for line in filename:
            if line[0:5] == struc_type:
                raw_line = line.split()
                chain = raw_line[chain_loc]
                locations = (int(raw_line[start]),int(raw_line[stop]))
                if chain in struc_dict:
                    if locations not in struc_dict[chain]:
                        struc_dict[chain].add(locations)
                else:
                    struc_dict[chain] = {(int(raw_line[start]),int(raw_line[stop]))}
                if chain not in chains:
                    chains.append(chain)
        return struc_dict, chains
    
    #getting secondary structures for each chain
    for structure_type in structure_types:
        structure, structure_chains = sec_struc_type(structure_type, pdb_file)
        for structure_chain in structure_chains:
            if structure_chain not in ss_by_struc:
                ss_by_struc[structure_chain] = {}
            ss_by_struc[structure_chain][structure_type] = structure[structure_chain]
            
    #getting metadata on the secondary structures
    ss_meta = {}
    for chain, ss_structures in ss_by_struc.items():
        if chain not in ss_meta:
            ss_meta[chain] = {}
        for ss_structure, ss_structure_contents in ss_structures.items():
            ss_meta[chain][ss_structure+'_MAX_LENGTH'] = max({id_stop-id_start+1 for id_start, id_stop in ss_structure_contents})
            ss_meta[chain][ss_structure+'_MIN_LENGTH'] = min({id_stop-id_start+1 for id_start, id_stop in ss_structure_contents})
            
            ss_meta[chain]['NUM_'+ss_structure] = len(ss_structure_contents)
            
    return ss_by_struc, ss_meta
