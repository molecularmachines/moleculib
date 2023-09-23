import esm
from colabdesign.mpnn import mk_mpnn_model


def recover_sequence(pdb_filepath, model='mpnn', chains=["A"], temp=0.1, num_samples=8):
    samples = None
    if model == 'proteinmpnn':
        mpnn_model = mk_mpnn_model()
        mpnn_model.prep_inputs(pdb_filename=pdb_filepath, chain=chain)
        samples = mpnn_model.sample_parallel(temperature=temp, batch=128)
    elif model == 'esm':
        esm_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        esm_model = esm_model.eval()
        structure = esm.inverse_folding.util.load_structure(pdb_filepath, chains)
        coords, samples = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
    return samples
