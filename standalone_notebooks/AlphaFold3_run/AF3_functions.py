import os
import subprocess
from Bio.PDB import PDBParser
import random
import json

def three_to_one(three_letter_name):
    """Converts three letter amino acid code to one letter code."""
    aa_dict = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
    return aa_dict.get(three_letter_name, 'X')  # Returns 'X' for unknown residues

def seeds_generator(seeds_number=3, predefined_seeds=False, predefined_seeds_list=None):
    """Generate seeds for AlphaFold3."""
    if predefined_seeds:
        seeds = predefined_seeds_list
    else:
        seeds = set()
        while len(seeds) < seeds_number:
            seed = random.randint(1, 128)
            seeds.add(seed)
        seeds = list(seeds)
    return seeds

def get_basic_sequences_dictionary (pdb_file):
    """Extracts sequence from a PDB file and returns a dictionary for AF3."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('X', pdb_file)
    protein_dicts = []
    
    model = next(structure.get_models())
    for chain in model:
        chain_id = chain.get_id()
        chain_dict = {}
        seq = ''
        for residue in chain:
            three_letter_name = residue.get_resname()
            one_letter_name = three_to_one(three_letter_name)
            seq += one_letter_name
        prot_entry = {
            "protein": {
                "id":       chain_id,
                "sequence": seq
                # Additionally, we could add PTM modifications, MSAs, templates, etc. here
            }
        }
        protein_dicts.append(prot_entry)
        
    return protein_dicts
        
    
def generate_json_from_pdb(pdb_file, json_folder, dialect, version, slurm_folder, account, seeds_number=3, predefined_seeds=False, predefined_seeds_list=None):
    # generate seeds_number seeds randomly 
    model_seeds = seeds_generator(seeds_number, predefined_seeds, predefined_seeds_list)
    name = os.path.basename(pdb_file).replace('.pdb', '')
    
    # generate json file for the input
    json_file_path = os.path.join(json_folder, f'{name}.json')
    protein_dicts = get_basic_sequences_dictionary(pdb_file)
    json_dict = {
        "name": name,
        "sequences": protein_dicts,
        "modelSeeds": model_seeds,
        "dialect": dialect,
        "version": version
    }
    with open(json_file_path, 'w') as f:
        json.dump(json_dict, f, indent=4)
    print(f'{json_file_path} json file generated')
    return json_file_path 
    
def prepare_slurm_script(pdb_input_file, output_folder, json_folder, dialect, version, slurm_folder, account, seeds_number=3, predefined_seeds=False, predefined_seeds_list=None):
    """Prepare a SLURM script for running AlphaFold3 on the input FASTA file."""

    input_json_file_path = generate_json_from_pdb(pdb_input_file, json_folder, dialect, version, slurm_folder, account, seeds_number, predefined_seeds, predefined_seeds_list)

    script_content = f'''#!/bin/bash
#SBATCH --job-name=AF3
#SBATCH --output={slurm_folder}/AF3_%j.out
#SBATCH --error={slurm_folder}/AF3_%j.err
#SBATCH --partition=gpu_mig40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:40:00
#SBATCH --mem=128GB
#SBATCH --account={account}

echo "CURRENT WORKING DIRECTORY = $(pwd)"

# Optimize GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # Disables memory preallocation
export TF_FORCE_UNIFIED_MEMORY=true             # Enables unified memory (if GPU RAM fills up it can spill into host CPU RAM)
export XLA_CLIENT_MEM_FRACTION=3.2              # Your job can request up to 3.2 times the GPU memory due to the unified memory setting
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false \
--xla_disable_hlo_passes=custom-kernel-fusion-rewriter" # There are known numerical issues with CUDA Capability 7.x devices. To work around the issue, set the ENV XLA_FLAGS to include --xla_disable_hlo_passes=custom-kernel-fusion-rewriter

# Load AF3
# Note that this version of alphafold is run in a Python (3.11) "venv" virtual environment. 
# The required cuda libraries are packaged with this version of alphafold so there is no need to load the cuda module to run alphafold on GPUs
module load Bioinformatics alphafold/3.0.0

# Run AF3
# example: python $AF3_SCRIPT_DIR/run_alphafold.py --json_path=$AF3_PARAMS_DIR/input/fold_input.json --output_dir=your_output_dir --db_dir=$AF3_PARAMS_DIR/db_dir --model_dir=$AF3_PARAMS_DIR/af3_model_params
python $AF3_SCRIPT_DIR/run_alphafold.py --json_path={input_json_file_path} --output_dir={output_folder} --db_dir=$AF3_PARAMS_DIR/db_dir --model_dir=$AF3_PARAMS_DIR/af3_model_params

module unload alphafold
'''

    name = os.path.basename(pdb_input_file).replace('.pdb', '')
    slurm_script_path = os.path.join(slurm_folder, f'{name}.sh')
    with open(slurm_script_path, 'w') as f:
        f.write(script_content)
    return slurm_script_path


def submit_af3_jobs(pdb_input_folder, AF_output_folder, json_folder, dialect, version, slurm_folder, account, seeds_number=3, predefined_seeds=False, predefined_seeds_list=None):
    """Generate and submit SLURM jobs for AlphaFold3 predictions.
            For each PDB file in the input folder, a corresponding JSON file is generated.
            Then, a SLURM script is created and submitted.
    """
    
    job_ids = []
    
    for pdb_file in os.listdir(pdb_input_folder):
        if not pdb_file.endswith('.pdb'):
            continue
        pdb_file_path = os.path.join(pdb_input_folder, pdb_file)
        
        output_folder = f'{AF_output_folder}/{pdb_file.replace(".pdb", "")}'
        os.makedirs(output_folder, exist_ok=True)
        
        # Prepare and submit the SLURM script
        print(f"Submitting {pdb_file_path}")
        slurm_script_path = prepare_slurm_script(pdb_file_path, output_folder, json_folder, dialect, version, slurm_folder, account, seeds_number, predefined_seeds, predefined_seeds_list)
        result = subprocess.run(["sbatch", slurm_script_path], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1] if result.returncode == 0 else None
        
        if job_id:
            print(f"AlphaFold3 job for {pdb_file} submitted with Job ID: {job_id}")
            job_ids.append(job_id)
    return job_ids