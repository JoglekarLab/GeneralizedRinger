import os
import subprocess
from multiprocessing import Pool
import pandas as pd
from job_manager import JobManager
import time

class DesignGeneratorPipeline:
    def __init__(self, nsym, radius, pyrosetta_env, general_env, account, temperature_to_seq_count, mpnn_output_base_path, source_folder_geometries_path, interface, fixed_chains, cb_distance, extended_design, max_distance):
        self.nsym = nsym
        self.radius = radius
        self.pyrosetta_env = pyrosetta_env
        self.general_env = general_env
        self.account = account
        self.temperature_to_seq_count = temperature_to_seq_count  # Dictionary of number of sequences to generate per temperature

        # Positions Generator Attributes
        self.interface = interface
        self.fixed_chains = fixed_chains
        self.cb_distance = cb_distance
        self.extended_design = extended_design
        self.max_distance = max_distance

        # Paths
        self.mpnn_output_base_path = mpnn_output_base_path  # Base path for MPNN design outputs
        self.source_folder_geometries = source_folder_geometries_path
        self.output_folder_geometry = f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/Outputs-GeometryAnalyzer' # f'../6_GenerateDesigns/{NSYM}mer/{NSYM}_r{RADIUS}/Outputs-GeometryAnalyzer' for 2nd iteration
        self.output_folder_proteinmpnn = f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/Outputs-ProteinMPNN'
        self.scores_folder = f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/OptimizationScores'
        self.mutated_pdbs_output = f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/OptimizedSequences'
        self.jobfile_path = f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/SlurmScripts/pyrosettamutate_jobfile'

        # Sequence folders for different temperatures
        self.sequence_folders = [f'{self.output_folder_proteinmpnn}/T_{key_temp}' for key_temp in self.temperature_to_seq_count.keys()]

    def copy_pdb_files(self):
        """Step 1: Copy PDB files to the output folder."""
        if not os.path.exists(self.output_folder_geometry) or len(os.listdir(self.output_folder_geometry)) == 0:
            os.makedirs(self.output_folder_geometry, exist_ok=True)
            os.system(f'cp {self.source_folder_geometries}/*.pdb {self.output_folder_geometry}')
        else:
            print(f"Folder {self.output_folder_geometry} already exists. Skipping copying pdb files.")

    def generate_designable_positions(self):
        """Step 2: Generate designable residues using pyRosetta."""
        
        extended_flag = "--extended_design" if self.extended_design else ""
        command = (
            f"python DesignPositionsGenerator.py "
            f"--pdb_dir {self.output_folder_geometry} "
            f"--interface {self.interface} "
            f"--fixed_chains {self.fixed_chains} "
            f"--cb_distance {self.cb_distance} "
            f"{extended_flag} "
            f"--max_distance {self.max_distance}"
        ).strip()


        slurm_script_name = f"run_3_1.sh"
        script_content = f"""#!/bin/bash
#SBATCH --job-name={self.nsym}{self.radius}DesPos
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=1:00:00
#SBATCH --account={self.account}

eval "$(conda shell.bash hook)"
conda activate {self.pyrosetta_env}

{command}
"""

        with open(slurm_script_name, 'w') as f:
            f.write(script_content)

        result = subprocess.run(["sbatch", slurm_script_name], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1] if result.returncode == 0 else None
        # os.remove(slurm_script_name)

        if job_id:
            print(f"Job for determining the positions in the interface to be redesigned submitted with Job ID: {job_id}", flush=True)
            JobManager.check_job_completion([job_id])
        
        return job_id

    def generate_mpnn_sequences(self, previous_job_id=None):
        """Step 3: Generate sequences using MPNN for different temperatures."""
        job_ids = [previous_job_id] if previous_job_id else []
        for temperature, seq_per_temp in self.temperature_to_seq_count.items():
            command = f"sbatch"
            if job_ids:
                command += f" --dependency=afterok:{job_ids[-1]}"
            command += f" SS_ProteinMPNNGenerator.sh {self.output_folder_geometry} {self.output_folder_proteinmpnn} {int(seq_per_temp)} {temperature}"
            output = subprocess.run(command, shell=True, capture_output=True, text=True).stdout
            job_id = output.strip().split()[-1]
            job_ids.append(job_id)
            print(f"Submitted job {job_id} with temperature {temperature} and seq_per_temp {seq_per_temp}", flush=True)
            time.sleep(5)
        JobManager.check_job_completion([job_id])
        return job_id

    def join_sequences(self):
        """Step 4: Join all sequences generated at different temperatures."""
        for file in os.listdir(self.sequence_folders[0]):
            for i in range(1, len(self.sequence_folders)):
                file_i = os.path.join(self.sequence_folders[i], file)
                with open(file_i, 'r') as f_i:
                    lines_i = f_i.readlines()
                    with open(os.path.join(self.output_folder_proteinmpnn, file), 'a') as f:
                        f.writelines(lines_i[2:])

    def extract_sequences(self, previous_job_id=None):
        """Step 5: Extract the top sequences using a Python script."""
        command = f'python Extract_TopMPNN.py --mpnn_folder {self.output_folder_proteinmpnn}'

        slurm_script_name = f"Extract.sh"
        script_content = f"""#!/bin/bash
#SBATCH --job-name={self.nsym}-{self.radius}
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --account={self.account}

eval "$(conda shell.bash hook)"
conda activate {self.general_env}

{command}
"""

        with open(slurm_script_name, 'w') as f:
            f.write(script_content)

        result = subprocess.run(["sbatch", slurm_script_name], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1] if result.returncode == 0 else None
        # os.remove(slurm_script_name)

        if job_id:
            print(f"Extract sequences SLURM job submitted with Job ID: {job_id}")
            JobManager.check_job_completion([job_id])

        return job_id

    @staticmethod
    def get_file_command(file, fasta_folder_path, structures_folder_path, scores_folder, mutated_pdbs_output):
        """Creates the command for mutating PDB files."""
        return f'python mutate_pdb.py --fasta_file_path {os.path.join(fasta_folder_path, file)} --structures_folder_path {structures_folder_path} --scores_folder {scores_folder} --output_folder {mutated_pdbs_output}'

    def get_file_command_wrapper(self, file):
        """Wrapper to call get_file_command with instance variables."""
        fasta_folder_path = f'{self.output_folder_proteinmpnn}/TopSequences'
        structures_folder_path = self.output_folder_geometry
        return self.get_file_command(file, fasta_folder_path, structures_folder_path, self.scores_folder, self.mutated_pdbs_output)

    def prepare_mutation_jobfile(self):
        """Step 6: Prepare the jobfile for PyRosetta mutation using fasta files."""
        fasta_folder_path = f'{self.output_folder_proteinmpnn}/TopSequences'
        structures_folder_path = self.output_folder_geometry
        
        os.makedirs(self.scores_folder, exist_ok=True)
        os.makedirs(self.mutated_pdbs_output, exist_ok=True)
        os.makedirs(os.path.dirname(self.jobfile_path), exist_ok=True)

        file_list = os.listdir(fasta_folder_path)

        with Pool() as pool:
            result_commands = pool.map(self.get_file_command_wrapper, file_list) # get_file_command_wrapper is a herper function not a class function so it is not called with self
        
        with open(self.jobfile_path, 'w') as f:
            for command in result_commands:
                f.write(command + '\n')

    def submit_mutation_job(self, previous_job_id=None):
        """Step 7: Submit the jobfile with SLURM."""
        script_content = f"""#!/bin/bash
#SBATCH --job-name={self.nsym}{self.radius}pyR
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=standard
#SBATCH --time=4:00:00
#SBATCH --ntasks=416
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=1
#SBATCH --account={self.account}

eval "$(conda shell.bash hook)"
conda activate {self.pyrosetta_env}

module load launcher
export LAUNCHER_JOB_FILE={self.jobfile_path}
$LAUNCHER_DIR/paramrun
"""

        slurm_script_name = f"run_pyrosetta_mutate_jobfile.sh"
        with open(slurm_script_name, 'w') as f:
            f.write(script_content)

        result = subprocess.run(["sbatch", slurm_script_name], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1] if result.returncode == 0 else None
        os.remove(slurm_script_name)

        if job_id:
            print(f"Mutation SLURM job submitted with Job ID: {job_id}")
            JobManager.check_job_completion([job_id])
        return job_id

    def run_pipeline(self):
        """Runs the entire design generator pipeline."""
        self.copy_pdb_files()
        design_job_id = self.generate_designable_positions()
        mpnn_job_id = self.generate_mpnn_sequences()
        self.join_sequences()
        extract_job_id = self.extract_sequences()
        self.prepare_mutation_jobfile()
        self.submit_mutation_job()
