import os
import argparse
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import re
import json
from Bio.PDB import PDBParser, Superimposer, MMCIFParser
from Bio import PDB, SeqIO
from Bio.PDB.Polypeptide import is_aa

def get_hydropathy_score (ala_count, arg_count, asn_count, asp_count, cys_count, gln_count, glu_count, gly_count, his_count, ile_count, leu_count, lys_count, met_count, phe_count, pro_count, ser_count, thr_count, trp_count, tyr_count, val_count):
    aa_scores = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
    total_residues = ala_count + arg_count + asn_count + asp_count + cys_count + gln_count + glu_count + gly_count + his_count + ile_count + leu_count + lys_count + met_count + phe_count + pro_count + ser_count + thr_count + trp_count + tyr_count + val_count
    hydropathy_score = (ala_count * aa_scores['A'] + arg_count * aa_scores['R'] + asn_count * aa_scores['N'] + asp_count * aa_scores['D'] + cys_count * aa_scores['C'] + gln_count * aa_scores['Q'] + glu_count * aa_scores['E'] + gly_count * aa_scores['G'] + his_count * aa_scores['H'] + ile_count * aa_scores['I'] + leu_count * aa_scores['L'] + lys_count * aa_scores['K'] + met_count * aa_scores['M'] + phe_count * aa_scores['F'] + pro_count * aa_scores['P'] + ser_count * aa_scores['S'] + thr_count * aa_scores['T'] + trp_count * aa_scores['W'] + tyr_count * aa_scores['Y'] + val_count * aa_scores['V'])/total_residues
    return hydropathy_score

def get_ca_rmsd(pdb1_path, pdb2_path):
    """
    Returns the C-alpha RMSD between two structures (PDB or CIF).
    Automatically picks the correct parser based on file extension.
    """
    if pdb1_path.lower().endswith(".cif"):
        parser1 = MMCIFParser(QUIET=True)
    else:
        parser1 = PDBParser(QUIET=True)
    struct1 = parser1.get_structure("initial", pdb1_path)
    
    if pdb2_path.lower().endswith(".cif"):
        parser2 = MMCIFParser(QUIET=True)
    else:
        parser2 = PDBParser(QUIET=True)
    struct2 = parser2.get_structure("designed", pdb2_path)
    
    pdb1_atoms = [atom for atom in struct1.get_atoms() if atom.name == "CA"]
    pdb2_atoms = [atom for atom in struct2.get_atoms() if atom.name == "CA"]
    if len(pdb1_atoms) != len(pdb2_atoms):
        raise ValueError("The structures have different numbers of C-alpha atoms.")
    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(pdb1_atoms, pdb2_atoms)
    return super_imposer.rms

def get_properties (df, file_id):
    dG_separated = df.loc[df['ID'] == file_id, 'dG_separated'].item()
    sc_value = df.loc[df['ID'] == file_id, 'sc_value'].item()
    dSASA_int = df.loc[df['ID'] == file_id, 'dSASA_int'].item()
    hydropathy_score = df.loc[df['ID'] == file_id, 'hydropathy_score'].item()
    hydrophobic_percentage = df.loc[df['ID'] == file_id, 'hydrophobic_percentage'].item()
    cluster_label = df.loc[df['ID'] == file_id, 'cluster_label'].item()
    total_geometry_score = df.loc[df['ID'] == file_id, 'total_geometry_score'].item()
    geometry_score = df.loc[df['ID'] == file_id, 'geometry_score'].item()
    n_contacts_geometry = df.loc[df['ID'] == file_id, 'n_contacts_geometry'].item()
    return dG_separated, sc_value, dSASA_int, hydropathy_score, hydrophobic_percentage, cluster_label, total_geometry_score, geometry_score, n_contacts_geometry

def extract_base_name(name):
    parts = name.split('_')
    base = []
    seen_rank = False

    for p in parts:
        if not seen_rank:
            # we haven't hit a rank... yet, so keep everything
            if p.startswith('rank'):
                seen_rank = True
            base.append(p)
        else:
            # once we've seen rank..., only keep further rank... segments
            if p.startswith('rank'):
                base.append(p)
            else:
                break

    return '_'.join(base)

def get_chain_lengths(cif_path):
    """
    Parse a CIF file and return a list with the (chain_id, residue_count)
    for every chain in the CIF file
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", cif_path)
    chain_info = []
    for model in structure:
        for chain in model:
            # count only standard amino‑acid residues
            length = sum(1 for res in chain if is_aa(res))
            chain_info.append((chain.id, length))
    return chain_info    

def get_pae_matrix (cif_path, json_path, chains_pairs_omit=set()):
    """
    Calculates the average pae for relevant pairs.
    IMPORTANT:  Note that the PAE(i,j) is the expected positional error when you align your entire model so that residue i
                sits exactly at its experimental location. Hence, PAE(i,j) != PAE(j,i), so both are calculated.
        Args:
        cif_path: Path to the model.cif file. Used to determine the length of chains (boundaries).
        json_path: Path to the confidences.json (AF3 output)
        chains_pairs_omit (optional):
            Pairs of chain IDs to skip when computing inter‑chain PAE, e.g. {('B', 'C'), ('C', 'B')}.

        Returns:
        boundaries: Dictionary of chain start and end, e.g. {'A': (0, 24), 'B': (24, 146), 'C': (146, 264)}
        pae_matrix: Square matrix (n_chains x n_chains) where entry [i,j] is the average PAE between chain i and chain j.
                    The diagonal is zeros
    """
    data    = json.load(open(json_path))
    pae = data["pae"]
    chain_info = get_chain_lengths(cif_path)
    assert len(pae) == sum(length for _, length in chain_info)
    pae = np.array(pae)

    boundaries = {}
    start = 0
    for chain_id, length in chain_info:
        end = start + length
        boundaries[chain_id] = (start, end)
        start = end

    pae_matrix = np.zeros((len(boundaries), len(boundaries)))
    for i, (chain_id_i, (bound_i)) in enumerate(boundaries.items()):
        for j, (chain_id_j, (bound_j)) in enumerate(boundaries.items()):
            if i != j: 
                if ((chain_id_i, chain_id_j) in chains_pairs_omit) or ((chain_id_j, chain_id_i) in chains_pairs_omit):
                    continue
                submatrix = pae[bound_i[0]:bound_i[1], bound_j[0]:bound_j[1]]
                avg_pae = np.mean(submatrix)
                pae_matrix[i][j] = avg_pae
                print(f"{chain_id_i} with {chain_id_j}")

    return boundaries, pae_matrix

def get_avg_ipTM (cif_path, summary_json_path, chains_pairs_omit=set()):
    '''
    Gets the interface predicted Template Modeling score (ipTM) which measures the accuracy of the predicted
    relative positions of the subunits forming the protein-protein complex. 
    Values higher than 0.8 represent confident high-quality predictions, while values below 0.6 suggest likely a failed prediction.
    ipTM values between 0.6 and 0.8 are a grey zone where predictions could be correct or wrong. 

    Disordered regions and regions with low pLDDT score may negatively impact the ipTM score even if the structure of the complex is predicted correctly.

    Note that the ipTM[i][j] == ipTM[j][i]
    '''

    data = json.load(open(summary_json_path))
    iptm = np.array(data["chain_pair_iptm"])
    chain_info = get_chain_lengths(cif_path)
    chain_ids = [cid for cid, _ in chain_info]

    values = []
    for i in range(len(chain_ids)):
        for j in range(i+1, len(chain_ids)):
            if i == j:
                continue
            if ((chain_ids[i], chain_ids[j]) in chains_pairs_omit) or ((chain_ids[j], chain_ids[i]) in chains_pairs_omit):
                continue
            values.append(iptm[i, j])
    return np.mean(values)

def get_pLDDT (cif_path, json_path, plddt_chains=[]):
    """
    Compute average pLDDT for specified chains or the whole structure.

    Args:
        plddt_chains (optional): List of chain IDs to include for calculating the mean pLDDT.

    Returns the mean pLDDT.
    """
    data = json.load(open(json_path))
    atom_plddts = data["atom_plddts"]
    
    if not plddt_chains: # An empty list is considered false
        return sum(atom_plddts) / len(atom_plddts) #avg

    chain_info = get_chain_lengths(cif_path)
    boundaries = {}
    start = 0
    for chain_id, length in chain_info:
        end = start + length
        boundaries[chain_id] = (start, end)
        start = end

    values = []
    for ch_id in plddt_chains:
        if ch_id not in boundaries:
            raise ValueError(f"Are you sure chain '{ch_id}' exists?")
        start, end = boundaries[ch_id]
        values.extend(atom_plddts[start:end])
    
    return sum(values) / len(values)


parser = argparse.ArgumentParser(description="Script to process AF2 predictions and analyze how well they match the designed geometries.")
parser.add_argument("--finished_predictions", required=True, help="Folder for storing finished prediction results, e.g., 'Finished'")
parser.add_argument("--pdb_input_folder", required=True, help="Folder containing initial PDB files, e.g., 'PDBInput'")

parser.add_argument(
    "--omit_pair",
    required=False,
    action="append", # Allows the user to input multiple pairs
    help=(
        "Chain pair to skip when computing PAE. Can be given multiple times, e.g. --omit_pair B,C --omit_pair C,D"
    )
)

parser.add_argument(
    "--plddt_chains",
    nargs="+", # Converts the user input --plddt_chains A B C into ["A","B","C"]
    required=False,
    default=[],
    help=(
        "Chain IDs to include in the pLDDT average (e.g. --plddt_chains A B C). "
        "If omitted, the average over all residues will be returned."
    )
)


args = parser.parse_args()

FinishedPredictions = args.finished_predictions
InitialGeometries = args.pdb_input_folder
plddt_chains = args.plddt_chains

chains_pairs_omit = set()
if args.omit_pair:
    for pair in args.omit_pair:
        c1, c2 = pair.split(",")
        chains_pairs_omit.add((c1, c2))
# This should look like {('A', 'B'), ('A', 'C')}


data_list = []
write_interval = 10
counter = 0

for pred_dir in os.listdir(FinishedPredictions):
    file_id = os.path.basename(pred_dir)
    pred_dir_path = os.path.join(FinishedPredictions, pred_dir)
    if not os.path.isdir(pred_dir_path):
        continue

    # creates a dictionary that assigns the AF_rank to the predictions depending on their ranking_score
    ranking_csv_path = os.path.join(pred_dir_path, "ranking_scores.csv")
    df_rank = pd.read_csv(ranking_csv_path)
    df_rank_sorted = df_rank.sort_values(by="ranking_score", ascending=False).reset_index(drop=True)
    df_rank_sorted["AF_rank"] = df_rank_sorted.index + 1 # Where AF rank = 1 is the highest scored
    AF_rank_dict = dict(zip(
        df_rank_sorted["sample"],
        zip(df_rank_sorted["AF_rank"], df_rank_sorted["ranking_score"])
        )
    )

    initial_name = extract_base_name(pred_dir)
    initial_structure_file = f"{InitialGeometries}/{initial_name}.pdb"

    for seed_dir in os.listdir(pred_dir_path):
        seed_path = os.path.join(pred_dir_path, seed_dir)
        if not os.path.isdir(seed_path):
            continue

        seed_number = seed_dir.split("seed-")[1].split("_")[0]
        sample_number = int(seed_dir.split("-")[2])
        seed_sample = f"{seed_number}-{sample_number}"

        confidences_json_path = os.path.join(pred_dir_path, seed_dir, "confidences.json")
        model_cif_path = os.path.join(pred_dir_path, seed_dir, "model.cif")
        summary_json_path = os.path.join(pred_dir_path, seed_dir, "summary_confidences.json")
        boundaries, pae_matrix = get_pae_matrix (model_cif_path, confidences_json_path, chains_pairs_omit)
        nonzero_vals = pae_matrix[pae_matrix != 0] #  get only the calculated entries of the matrix
        iPAE = nonzero_vals.mean()
        ipTM = get_avg_ipTM(model_cif_path, summary_json_path, chains_pairs_omit)
        pLDDT = get_pLDDT (model_cif_path, confidences_json_path, plddt_chains)
        (AF_rank, ranking_score) = AF_rank_dict[sample_number]

        rmsd = get_ca_rmsd(initial_structure_file, model_cif_path)
        geometry = pred_dir.split('_rank')[0]

        data_list.append({'ID': file_id, 'AF_rank': AF_rank, 'average_plddt': pLDDT, 'ipTM': ipTM, 'iPAE': iPAE, 'rmsd': rmsd, 'ranking_score': ranking_score, 'geometry': geometry})
    
    counter += 1
    if counter % write_interval == 0:
        print(f'Processed {counter} directories, writing intermediate results...')
        df = pd.DataFrame(data_list)
        df.to_csv(f'{FinishedPredictions}/AF_results_intermediate.csv', index=False)
        print(f'Intermediate results saved to {FinishedPredictions}/AF_results_intermediate.csv')
        
df = pd.DataFrame(data_list)

df.to_csv(f'{FinishedPredictions}/AF_results.csv', index=False)
