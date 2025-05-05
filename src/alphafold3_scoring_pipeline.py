import os
import subprocess
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser
from job_manager import JobManager
import random
import json
import shutil

class AlphaFold_3_ScoringPipeline:
    def __init__(self, nsym, radius, account, general_env, pdb_input_folder_path, AF_folder, seeds_number=3, predefined_seeds=False, predefined_seeds_list=None, dialect="alphafold3", version="3", pae_omit_pairs=None, plddt_chains=None):
        self.nsym = nsym
        self.radius = radius
        self.account = account
        self.general_env = general_env

        # Paths        
        self.AF_output_folder = f'{AF_folder}/{nsym}mer/{nsym}_r{radius}/Outputs'
        self.slurm_folder = f'{AF_folder}/{nsym}mer/{nsym}_r{radius}/SlurmScripts'
        self.json_folder = f'{AF_folder}/{nsym}mer/{nsym}_r{radius}/JsonInputs'
        self.finished_predictions = f'{AF_folder}/{nsym}mer/{nsym}_r{radius}/Finished_AF_predictions'
        self.pdb_input_folder = pdb_input_folder_path
        
        # AF3-specific configurations
        
        # IMPORTANT! Note that the AF3 config is hardcoded in the $AF3_SCRIPT_DIR/run_alphafold.py!
        # Hence, some parameters cannot be straightforwardly changed, such as the number of recycles, which is set to 10.
        # For more information, check the model_config.json and run_alphafold.py in the AF3 documentation.
        
        self.dialect = dialect
        self.version = version
        self.seeds_number = seeds_number
        self.predefined_seeds = predefined_seeds
        
        if self.predefined_seeds:
            assert predefined_seeds_list is not None, "Predefined seeds list must be provided if predefined_seeds is True"
            self.predefined_seeds_list = predefined_seeds_list
        else:
            self.predefined_seeds_list = None
        
        # Parameters for analyzing AF results
        self.pae_omit_pairs = pae_omit_pairs or []
        self.plddt_chains   = plddt_chains   or []

        # Create required directories
        self.create_folders([self.AF_output_folder, self.slurm_folder, self.finished_predictions, self.json_folder])

    def create_folders(self, folder_list):
        """Create necessary folders if they don't exist."""
        for folder in folder_list:
            os.makedirs(folder, exist_ok=True)

    def three_to_one(self, three_letter_name):
        """Converts three letter amino acid code to one letter code."""
        aa_dict = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
        return aa_dict.get(three_letter_name, 'X')  # Returns 'X' for unknown residues

    def get_sequence(self, pdb_file):
        """Extracts sequence from a PDB file."""
        parser = PDBParser()
        structure = parser.get_structure('X', pdb_file)
        sequence = []
        for model in structure:
            for chain in model:
                seq = ''
                for residue in chain:
                    three_letter_name = residue.get_resname()
                    one_letter_name = self.three_to_one(three_letter_name)
                    seq += one_letter_name
                sequence.append(seq)
        return sequence

    def pdb_to_fasta(self, pdb_file):
        """Converts PDB file to FASTA format."""
        sequence = self.get_sequence(pdb_file)
        sequence_str = ':\n'.join(sequence)
        header = os.path.basename(pdb_file).replace('.pdb', '')
        fasta = f'>{header}\n{sequence_str}'
        return fasta
    
    def seeds_generator(self):
        """Generate seeds for AlphaFold3."""
        if self.predefined_seeds:
            seeds = self.predefined_seeds_list
        else:
            seeds = set()
            while len(seeds) < self.seeds_number:
                seed = random.randint(1, 128)
                seeds.add(seed)
            seeds = list(seeds)
        return seeds
    
    def get_basic_sequences_dictionary (self, pdb_file):
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
                one_letter_name = self.three_to_one(three_letter_name)
                seq += one_letter_name
            
            no_msa = f">query\n{seq}\n"
            prot_entry = {
                "protein": {
                    "id":       chain_id,
                    "sequence": seq,
                    # Additionally, we could add PTM modifications, MSAs, templates, etc. here
                    "modifications": [],
                    "unpairedMsa": no_msa, # Can change it for a precomputed MSA
                    "pairedMsa": no_msa,
                    "templates": []
                }
            }
            protein_dicts.append(prot_entry)
            
        return protein_dicts
            
        
    def generate_json_from_pdb(self, pdb_file):
        # generate seeds_number seeds randomly 
        model_seeds = self.seeds_generator()
        name = os.path.basename(pdb_file).replace('.pdb', '')
        
        # generate json file for the input
        json_file_path = os.path.join(self.json_folder, f'{name}.json')
        protein_dicts = self.get_basic_sequences_dictionary(pdb_file)
        json_dict = {
            "name": name,
            "sequences": protein_dicts,
            "modelSeeds": model_seeds,
            "dialect": self.dialect,
            "version": self.version
        }
        with open(json_file_path, 'w') as f:
            json.dump(json_dict, f, indent=4)
        print(f'{json_file_path} json file generated', flush=True)
        return json_file_path 

    def prepare_slurm_script(self, pdb_input_file, output_folder):
        """Prepare a SLURM script for running AlphaFold3 on the input FASTA file."""
        
        input_json_file_path = self.generate_json_from_pdb(pdb_input_file)
        
        script_content = f'''#!/bin/bash
#SBATCH --job-name=AF3
#SBATCH --output={self.slurm_folder}/AF3_%j.out
#SBATCH --error={self.slurm_folder}/AF3_%j.err
#SBATCH --partition=gpu_mig40,gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:15:00
#SBATCH --mem=12GB
#SBATCH --account={self.account}

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
        slurm_script_path = os.path.join(self.slurm_folder, f'{name}.sh')
        with open(slurm_script_path, 'w') as f:
            f.write(script_content)
        return slurm_script_path

    def submit_af3_jobs(self):
        """
        Generate and submit SLURM jobs for AlphaFold3 predictions.
        For each PDB file in the input folder, a corresponding JSON file is generated.
        Then, a SLURM script is created and submitted.
        """
        
        job_ids = []
        
        for pdb_file in os.listdir(self.pdb_input_folder):
            if not pdb_file.endswith('.pdb'):
                continue
            
            pdb_file_path = os.path.join(self.pdb_input_folder, pdb_file)
            output_folder = f'{self.AF_output_folder}/{pdb_file.replace(".pdb", "")}'
            os.makedirs(output_folder, exist_ok=True)
            
            # Prepare and submit the SLURM script
            slurm_script_path = self.prepare_slurm_script(pdb_file_path, output_folder)
            result = subprocess.run(["sbatch", slurm_script_path], capture_output=True, text=True)
            job_id = result.stdout.strip().split()[-1] if result.returncode == 0 else None
            
            if job_id:
                print(f"AlphaFold3 job for {pdb_file} submitted with Job ID: {job_id}", flush=True)
                job_ids.append(job_id)
                
            # Remove the SLURM script after submission
            # os.remove(slurm_script_path)
        JobManager.check_job_completion(job_ids)

    def monitor_af_predictions(self):
        """Monitor AlphaFold predictions and move completed ones."""
        n = 0
        for output_dir in os.listdir(self.AF_output_folder):
            output_dir_path = os.path.join(self.AF_output_folder, output_dir)
            if not os.path.isdir(output_dir_path):
                continue

            if len(os.listdir(output_dir_path)) > 25:
                print(f"{output_dir_path} completed")
                n += 1
                # Move to Finished Predictions
                os.system(f'mv {output_dir_path} {self.finished_predictions}')
            else:
                print(f"{output_dir_path} has less than 25 predictions")

        print(f"Number of completed predictions: {n}")

    def monitor_af3_predictions(self):
        """Monitor AlphaFold predictions and move completed ones."""
        n = 0
        for output_dir in os.listdir(self.AF_output_folder):
            output_dir_path = os.path.join(self.AF_output_folder, output_dir)
            if not os.path.isdir(output_dir_path):
                continue

            if len(os.listdir(output_dir_path)) > 25:
                print(f"{output_dir_path} completed")
                n += 1
                # Move to Finished Predictions
                os.system(f'mv {output_dir_path} {self.finished_predictions}')
            else:
                print(f"{output_dir_path} has less than 25 predictions")

        print(f"Number of completed predictions: {n}")


    def monitor_af3_predictions(self):
        os.makedirs(self.finished_predictions, exist_ok=True)

        for output_dir in os.listdir(self.AF_output_folder):
            output_dir_path = os.path.join(self.AF_output_folder, output_dir)
            if not os.path.isdir(output_dir_path):
                continue

            for subrun in os.listdir(output_dir_path):
                subrun_path = os.path.join(output_dir_path, subrun)
                if not os.path.isdir(subrun_path):
                    continue

                if len(os.listdir(subrun_path)) > 6:
                    shutil.move(subrun_path, self.finished_predictions)

    def analyze_af_predictions(self, rosetta_file_path, designs_scores):
        """Analyze AlphaFold predicted structures using AF_process_results.py."""
        command = f'python AF_process_results.py --rosetta_file_path {rosetta_file_path} --finished_predictions {self.finished_predictions} --pdb_input_folder {self.pdb_input_folder} --designs_scores {designs_scores}'

        if self.pae_omit_pairs:
            for c1, c2 in self.pae_omit_pairs:
                command += f' --omit_pair {c1},{c2}'

        if self.plddt_chains:
            command += ' --plddt_chains ' + ' '.join(self.plddt_chains)

        slurm_script_name = f"AF_process_results.sh"
        script_content = f"""#!/bin/bash
#SBATCH --job-name={self.nsym}-{self.radius}AF
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --account={self.account}

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate {self.general_env}

# Run the python script
{command}
"""
        # Write SLURM script and submit it
        with open(slurm_script_name, 'w') as f:
            f.write(script_content)
        
        result = subprocess.run(["sbatch", slurm_script_name], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1] if result.returncode == 0 else None

        if job_id:
            print(f"AlphaFold process analysis SLURM job submitted with Job ID: {job_id}")
            from job_manager import JobManager
            JobManager.check_job_completion([job_id])
        else:
            print("Failed to submit analysis SLURM job")

    def inspect_results(self, rmsd_threshold=3.5):
        """Inspect and process the final AlphaFold results."""
        AF_results_path = f"{self.finished_predictions}/AF_results.csv"
        print (f'Max RMSD to target geometry is {rmsd_threshold} Å')
        df_AF_results = pd.read_csv(AF_results_path)
        df_AF_results = df_AF_results[df_AF_results['rmsd'] < rmsd_threshold]
        df_AF_results.sort_values(by='ipTM', ascending=False, inplace=True)
        if df_AF_results['rmsd'].isnull().any():
            print("**** WARNING: Some rows have NaN values. ****")
        return df_AF_results

    def run_pipeline(self, rosetta_file_path, designs_scores, rmsd_threshold=3.5):
        """Runs the entire AlphaFold prediction and analysis pipeline."""
        
        print("Submitting AlphaFold jobs...")
        self.submit_af3_jobs()
        self.monitor_af3_predictions()
        print("Analyzing AlphaFold predicted structures...")
        self.analyze_af_predictions(rosetta_file_path, designs_scores)
        results = self.inspect_results(rmsd_threshold)
        print("AlphaFold pipeline completed. For all results see the generated csv file. Here are the top results:")
        print(results.head(10))
        AF_select_results_path = f"{self.finished_predictions}/AF_top_results.csv"
        results.to_csv(AF_select_results_path)
        print("Saved results with low RMSD with target structure.")
        return results