import os
import subprocess
import time
import pandas as pd
from job_manager import JobManager


class GeometryOptimizationPipeline:
    def __init__(self, nsym, radius, monomer_pdb_path, methods, pyrosetta_env, general_env, account, N=500):
        self.nsym = nsym
        self.radius = radius
        self.monomer_pdb_path = monomer_pdb_path
        self.methods = methods
        self.pyrosetta_env = pyrosetta_env
        self.general_env = general_env
        self.account = account
        self.N = N  # Maximum number of top scoring. geometry files to select for further processing, default is 500
        
        # Paths
        self.folder_name = f"Fixed_nSym{nsym}_radius{radius}"
        self.output_dir = f"../1_Geometry_Selection/{nsym}mer/{nsym}_r{radius}/Outputs-GeometryAnalizer/{os.path.basename(monomer_pdb_path).replace('.pdb', '')}"
        self.geometry_folder_path = f'{self.output_dir}/{self.folder_name}'
        self.scores_folder_path = f'{self.output_dir}/Scores/{os.path.basename(monomer_pdb_path).replace(".pdb", "")}'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_slurm_script(self, method):
        """Generates the SLURM script for a given method."""
        slurm_script_name = f"run_{method}_{self.nsym}_{self.radius}.sh"
        script_content = f"""#!/bin/bash
#SBATCH --job-name={self.radius}_{self.nsym}-{method}
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=8:00:00
#SBATCH --account={self.account}

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate {self.general_env}

# Run the python script with the specified method
python GeometryOptimization.py --nSym {self.nsym} --radius {self.radius} --method {method} --monomer_pdb_path {self.monomer_pdb_path} --output_dir {self.output_dir}
"""
        with open(slurm_script_name, 'w') as f:
            f.write(script_content)
        return slurm_script_name
    
    def run_slurm_scripts(self):
        """Submits the SLURM scripts for all methods."""
        job_ids = []
        for method in self.methods:
            slurm_script_name = self.create_slurm_script(method)
            result = subprocess.run(["sbatch", slurm_script_name], capture_output=True, text=True)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                job_id = output.split()[-1]
                job_ids.append(job_id)
            else:
                raise RuntimeError(f"Failed to submit job: {result.stderr}")
            os.remove(slurm_script_name)  # Remove the SLURM script after submission
        return job_ids
    
    def join_results(self):
        """Joins results from all methods into one file."""
        header_written = False
        combined_file = os.path.join(self.geometry_folder_path, "geometry_total_scores_long.csv")

        for subfolder in os.listdir(self.geometry_folder_path):
            subfolder_path = os.path.join(self.geometry_folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith(".csv"):
                        with open(os.path.join(subfolder_path, file), 'r') as f:
                            lines = f.readlines()
                            if not header_written:
                                header_written = True
                                with open(combined_file, 'a') as output_file:
                                    output_file.writelines(lines)
                            else:
                                with open(combined_file, 'a') as output_file:
                                    output_file.writelines(lines[1:])
        print(f"Results joined into {combined_file}")
    
    def generate_pdb_files(self):
        """Generates PDB files using TopGeometryPDBGenerator. Generates dimers by default."""
        geometry_folder_path = self.geometry_folder_path
        scores_folder_path = self.scores_folder_path

        # Commands for generating dimers and whole rings
        dimer_output_folder = os.path.join(geometry_folder_path, "2-Dimers")
        os.makedirs(dimer_output_folder, exist_ok=True)
        command_dimer = f"python TopGeometryPDBGenerator.py --monomer_pdb_path {self.monomer_pdb_path} --geometry_folder_path {geometry_folder_path} --output_folder {dimer_output_folder} --scores_folder_path {scores_folder_path} --N {self.N}"
        
        whole_output_folder = os.path.join(geometry_folder_path, "Whole_Rings")
        command_whole_ring = f"python TopGeometryPDBGenerator.py --monomer_pdb_path {self.monomer_pdb_path} --geometry_folder_path {geometry_folder_path} --output_folder {whole_output_folder} --whole_ring --N {self.N}"

        slurm_script_name = f"run_generate_pdb.sh"
        script_content = f"""#!/bin/bash
#SBATCH --job-name={self.nsym}{self.radius}PDB
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=4:00:00
#SBATCH --account={self.account}

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate {self.pyrosetta_env}

# Generate dimer PDB files
{command_dimer}
"""

        with open(slurm_script_name, 'w') as f:
            f.write(script_content)

        result = subprocess.run(["sbatch", slurm_script_name], capture_output=True, text=True)
        job_id = None
        if result.returncode == 0:
            output = result.stdout.strip()
            job_id = output.split()[-1]
            print(f"PDB generation SLURM script submitted with Job ID: {job_id}")
        return [job_id]
        
    def run_clustering(self):
        """Runs the geometry clustering step by generating and submitting a SLURM script."""
        
        # Paths for the clustering step
        input_geometries_folder = os.path.join(self.geometry_folder_path, "2-Dimers")
        output_cluster_folder = f"../2_Geometry_Clustering/{self.nsym}mer/{self.nsym}_r{self.radius}"
        scores_file = f"../1_Geometry_Selection/{self.nsym}mer/{self.nsym}_r{self.radius}/Outputs-GeometryAnalizer/{os.path.basename(self.monomer_pdb_path).replace('.pdb', '')}/Scores/{os.path.basename(self.monomer_pdb_path).replace('.pdb', '')}/{self.folder_name}.csv"

        # Command for geometry clustering
        command = f"python geometry_clustering.py --input_geometries_folder {input_geometries_folder} --output_cluster_folder {output_cluster_folder} --scores_file {scores_file}"

        # SLURM script content for clustering
        slurm_script_name = f"run_GC.sh"
        script_content = f"""#!/bin/bash
#SBATCH --job-name={self.nsym}{self.radius}GC
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=6:00:00
#SBATCH --account={self.account}

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate {self.general_env}

# Run the geometry clustering script
{command}
"""

        with open(slurm_script_name, 'w') as f:
            f.write(script_content)
        result = subprocess.run(["sbatch", slurm_script_name], capture_output=True, text=True)
        os.remove(slurm_script_name)
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"Geometry Clustering SLURM script submitted with Job ID: {job_id}")
            JobManager.check_job_completion([job_id])
        else:
            raise RuntimeError(f"Failed to submit clustering job: {result.stderr}")
    
    def run_pipeline(self):
        # # 1. Geometry Optimization
        # job_ids = self.run_slurm_scripts()      
        # JobManager.check_job_completion(job_ids)  
        # self.join_results()
        
        # combined_file = os.path.join(self.geometry_folder_path, "geometry_total_scores_long.csv")
        # df = pd.read_csv(combined_file)
        # df = df[df['nSym'] == self.nsym]
        # df = df.sort_values(by=['score*n_contacts'], ascending=False)
        # df = df.drop_duplicates(subset=['radius', 'nSym', 'rotx', 'roty', 'rotz'], keep='first')
        # df['radius'] = df['radius'].apply(lambda x: float(x))
        # unselected_file = os.path.join(self.geometry_folder_path, "geometry_total_scores_unselected.csv")
        # selected_file = os.path.join(self.geometry_folder_path, "geometry_total_scores.csv")
        # df.to_csv(unselected_file, index=False)
        # df.to_csv(selected_file, index=False)
        # print(f"Processed results saved to {selected_file}")
        
        # # 2. Generate best PDB files
        # pdb_job_ids = self.generate_pdb_files()
        # JobManager.check_job_completion(pdb_job_ids)

        # 3. Cluster and select files
        self.run_clustering()
        print('Geometry pipeline finished.')