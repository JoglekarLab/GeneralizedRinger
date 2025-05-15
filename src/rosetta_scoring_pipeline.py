import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from job_manager import JobManager

class RosettaScoringPipeline:
    def __init__(self, nsym, radius, pyrosetta_env, account, mpnn_output_base_path, clustering_base_path, FastRelax_protocol_path, rosetta_exec_path):

        self.nsym = nsym
        self.radius = radius
        self.pyrosetta_env = pyrosetta_env
        self.account = account
        self.FastRelax_protocol_path = FastRelax_protocol_path
        self.rosetta_exec_path = rosetta_exec_path

        # Paths
        self.mpnn_output_base_path = mpnn_output_base_path
        self.clustering_base_path = clustering_base_path

        self.mutated_pdbs_output = f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/OptimizedSequences'
        self.rosetta_relaxed_structures = f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/RelaxedSequences'
        self.relaxed_optimization_scores = f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/RelaxedOptimizationScores'
        self.rosetta_score_output_file = f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/RelaxedOptimizationScores.csv'
        self.geometries_scores_file = f'{self.clustering_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/SelectedGeometries_Scores.csv'
        self.jobfile_path = f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/SlurmScripts/rosetta_jobfile'
        self.plots_folder = f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/Plots'

        os.makedirs(self.rosetta_relaxed_structures, exist_ok=True)
        os.makedirs(self.relaxed_optimization_scores, exist_ok=True)
        os.makedirs(os.path.dirname(self.jobfile_path), exist_ok=True)
        os.makedirs(self.plots_folder, exist_ok=True)

    def create_rosetta_jobfile(self):
        """Step 1: Create the jobfile for Rosetta FastRelax and InterfaceAnalyzer."""
        file_list = os.listdir(self.mutated_pdbs_output)
        commands_list = []

        for file in file_list:
            if not os.path.exists(os.path.join(self.rosetta_relaxed_structures, file.replace('.pdb', '_0001.pdb'))):
                # Command to run the Rosetta FastRelax protocol
                script_content = (
                    f'{self.rosetta_exec_path} '
                    f'-s {os.path.join(self.mutated_pdbs_output, file)} '
                    f'-corrections::beta_nov16 true '
                    f'-parser:protocol {self.FastRelax_protocol_path} '
                    f'-out:prefix {self.rosetta_relaxed_structures}/ '
                    f'-out:file:scorefile {os.path.join(self.relaxed_optimization_scores, file.replace(".pdb", ".csv"))}\n'
                )
                commands_list.append(script_content)

        with open(self.jobfile_path, 'w') as f:
            for command in commands_list:
                f.write(command)

    def submit_rosetta_jobfile(self):
        # Depending on the number of lines in the jobfile_path, select how many ntasks to run
        with open(self.jobfile_path, 'r') as jf:
            num_lines = sum(1 for _ in jf)
        ntasks = min(num_lines, 128)

        script_content = f"""#!/bin/bash
#SBATCH --job-name={self.nsym}{self.radius}FR
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=standard
#SBATCH --time=2:00:00
#SBATCH --ntasks={ntasks}
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=1
#SBATCH --account={self.account}

eval "$(conda shell.bash hook)"
conda activate {self.pyrosetta_env}

# Execute launcher
module load launcher
export LAUNCHER_JOB_FILE={self.jobfile_path}
$LAUNCHER_DIR/paramrun
"""

        slurm_script_name = "run_rosetta_relax_jobfile.sh"
        with open(slurm_script_name, 'w') as f:
            f.write(script_content)

        result = subprocess.run(["sbatch", slurm_script_name], capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"Rosetta FastRelax and InterfaceAnalyzer submitted with Job ID: {job_id}")
        os.remove(slurm_script_name)
        JobManager.check_job_completion([job_id])

    def concatenate_rosetta_scores(self):
        """Step 3: Concatenate the Rosetta scores into a single file."""
        with open(self.rosetta_score_output_file, 'w') as f:
            files = os.listdir(self.relaxed_optimization_scores)
            first_file = os.path.join(self.relaxed_optimization_scores, files[0])
            with open(first_file, 'r') as f1:
                f1.readline()
                line = f1.readline()
                f.write(line)
            for file in files:
                if file.endswith('.csv' or '.sc'):
                    file_path = os.path.join(self.relaxed_optimization_scores, file)
                    with open(file_path, 'r') as f2:
                        f2.readline()
                        f2.readline()
                        line = f2.readline()
                        f.write(line)

    @ staticmethod
    def get_hydropathy_score (ala_count, arg_count, asn_count, asp_count, cys_count, gln_count, glu_count, gly_count, his_count, ile_count, leu_count, lys_count, met_count, phe_count, pro_count, ser_count, thr_count, trp_count, tyr_count, val_count):
        aa_scores = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
        total_residues = ala_count + arg_count + asn_count + asp_count + cys_count + gln_count + glu_count + gly_count + his_count + ile_count + leu_count + lys_count + met_count + phe_count + pro_count + ser_count + thr_count + trp_count + tyr_count + val_count
        hydropathy_score = (ala_count * aa_scores['A'] + arg_count * aa_scores['R'] + asn_count * aa_scores['N'] + asp_count * aa_scores['D'] + cys_count * aa_scores['C'] + gln_count * aa_scores['Q'] + glu_count * aa_scores['E'] + gly_count * aa_scores['G'] + his_count * aa_scores['H'] + ile_count * aa_scores['I'] + leu_count * aa_scores['L'] + lys_count * aa_scores['K'] + met_count * aa_scores['M'] + phe_count * aa_scores['F'] + pro_count * aa_scores['P'] + ser_count * aa_scores['S'] + thr_count * aa_scores['T'] + trp_count * aa_scores['W'] + tyr_count * aa_scores['Y'] + val_count * aa_scores['V'])/total_residues
        return hydropathy_score
    
    def analyze_rosetta_scores(self):
        """Step 4: Analyze the Rosetta scores and plot distributions."""
        df_rosetta = pd.read_csv(self.rosetta_score_output_file, sep='\s+')
        df_rosetta.drop(df_rosetta['SCORE:'], axis=1, inplace=True)
        df_rosetta['description'] = df_rosetta['description'].astype(str)
        df_rosetta = df_rosetta.dropna()

        numeric_columns = [
            'ala_count', 'arg_count', 'asn_count', 'asp_count', 'cys_count', 'gln_count', 'glu_count', 'gly_count', 'his_count',
            'ile_count', 'leu_count', 'lys_count', 'met_count', 'phe_count', 'pro_count', 'ser_count', 'thr_count',
            'trp_count', 'tyr_count', 'val_count'
        ]

        for col in numeric_columns:
            df_rosetta[col] = pd.to_numeric(df_rosetta[col], errors='coerce')

        df_rosetta['geometry'] = [os.path.basename(x).split('_rank')[0] for x in df_rosetta['description']]
        df_rosetta['MPNN_rank'] = [int(os.path.basename(x).split('_rank')[-1].split('_')[0]) for x in df_rosetta['description']]
        df_rosetta['ID'] = [os.path.basename(x).replace('.pdb','').replace('_0001','') for x in df_rosetta['description']]
        df_rosetta['hydrophobic_percentage'] = df_rosetta['dSASA_hphobic'] / df_rosetta['dSASA_int']
        df_rosetta['polar_percentage'] = df_rosetta['dSASA_polar'] / df_rosetta['dSASA_int']
        df_rosetta['dG_dSASA'] = df_rosetta['dG_separated'] / df_rosetta['dSASA_int'] * 100

        df_rosetta['hydropathy_score'] = [self.get_hydropathy_score(
            x['ala_count'], x['arg_count'], x['asn_count'], x['asp_count'], x['cys_count'], 
            x['gln_count'], x['glu_count'], x['gly_count'], x['his_count'], x['ile_count'], 
            x['leu_count'], x['lys_count'], x['met_count'], x['phe_count'], x['pro_count'], 
            x['ser_count'], x['thr_count'], x['trp_count'], x['tyr_count'], x['val_count']
        ) for i, x in df_rosetta.iterrows()]

        print('Number of geometries dropped due to positive dG_separated: ', df_rosetta[df_rosetta['dG_separated'] > 0].shape[0])
        df_rosetta = df_rosetta[df_rosetta['dG_separated'] < 0]
        df_rosetta['description'] = [x+'.pdb' for x in df_rosetta['description']]
        
        df_rosetta_aa_composition = df_rosetta[['ala_count', 'arg_count', 'asn_count', 'asp_count', 'cys_count', 'gln_count', 'glu_count', 'gly_count', 'his_count', 'ile_count', 'leu_count', 'lys_count', 'met_count', 'phe_count', 'pro_count', 'ser_count', 'thr_count', 'trp_count', 'tyr_count', 'val_count', 'hydrophobic_count', 'polar_uncharged_count', 'charged_count']]
        df_rosetta_main = df_rosetta[['ID', 'dG_separated', 'dG_dSASA', 'hbonds_int', 'sc_value', 'nres_int', 'hydrophobic_percentage', 'polar_percentage', 'hydropathy_score', 'ala_count', 'dSASA_hphobic', 'dSASA_polar', 'dSASA_int', 'description', 'geometry']]

        df_rosetta.to_csv(f'{self.rosetta_relaxed_structures}/Rosetta_Scores_Analyzed.csv', index=False)
        self.plot_rosetta_analysis(df_rosetta)
        return df_rosetta 

    def plot_rosetta_analysis(self, df_rosetta):
        """Step 5: Generate plots from the Rosetta analysis."""

        # Plot the distribution of the ddG per interface SASA (dG_dSASA)
        plt.figure(figsize=(10, 6))
        sns.histplot(df_rosetta['dG_dSASA'], bins=100, color='darkblue')
        plt.xlabel('ddG/dSASA')
        plt.ylabel('Frequency')
        plt.title('Distribution of ddG/dSASA')
        plt.savefig(os.path.join(self.plots_folder, 'ddG_dSASA_distribution.png'))

        # Plot the fraction of polar residues in the interface
        plt.figure(figsize=(10, 6))
        sns.histplot(df_rosetta['polar_percentage'], bins=100, color='darkblue')
        plt.xlabel('Polar fraction')
        plt.ylabel('Count')
        plt.title('Fraction of polar residues in the interface')
        plt.savefig(os.path.join(self.plots_folder, 'polar_fraction.png'))

        # Plot the number of alanine (A) residues in the interface
        plt.figure(figsize=(10, 6))
        sns.histplot(df_rosetta['ala_count'], bins=100, color='darkblue')
        plt.xlabel('Number of alanine residues')
        plt.ylabel('Count')
        plt.title('Number of alanine residues in the interface')
        plt.savefig(os.path.join(self.plots_folder, 'alanine_count.png'))

        # Plot the number of interface H Bonds being estimated by Rosetta
        plt.figure(figsize=(10, 6))
        sns.histplot(df_rosetta['hbonds_int'], bins=100, color='darkblue')
        plt.xlabel('Number of interface H Bonds')
        plt.ylabel('Count')
        plt.title('Number of interface H Bonds')
        plt.savefig(os.path.join(self.plots_folder, 'hbonds_int.png'))

        # Plot the total binding energy (dG_separated) estimated by Rosetta
        plt.figure(figsize=(10, 6))
        sns.histplot(df_rosetta['dG_separated'], bins=100, color='darkblue')
        plt.xlabel('Total binding energy (dG_separated)')
        plt.ylabel('Count')
        plt.title('Total binding energy (dG_separated)')
        plt.savefig(os.path.join(self.plots_folder, 'dG_separated.png'))

        # Plot the interface shape complementarity (sc_value) estimated by Rosetta
        plt.figure(figsize=(10, 6))
        sns.histplot(df_rosetta['sc_value'], bins=100, color='purple')
        plt.xlabel('Shape Complementarity value')
        plt.ylabel('Count')
        plt.title('Shape Complementarity value')
        plt.savefig(os.path.join(self.plots_folder, 'sc_value.png'))

        # Plot the number of residues in the interface
        plt.figure(figsize=(10, 6))
        sns.histplot(df_rosetta['nres_int'], bins=100, color='darkblue')
        plt.xlabel('Number of interface residues')
        plt.ylabel('Count')
        plt.title('Number of interface residues')
        plt.savefig(os.path.join(self.plots_folder, 'nres_int.png'))

        # sc_value highest value quartile
        sc_value_quartile = df_rosetta['sc_value'].quantile(0.75)
        # dG_dSASA lowest value quartile
        dG_dSASA_quartile = df_rosetta['dG_dSASA'].quantile(0.25)

        # Plot of the correlation of ddG/dSASA with sc_value
        plt.figure(figsize=(10, 6))
        plt.scatter(df_rosetta['sc_value'], df_rosetta['dG_dSASA'], c=df_rosetta['nres_int'])
        plt.colorbar(label='Number of interface residues')
        plt.xlabel('Shape Complementarity value')
        plt.ylabel('ddG/dSASA')
        plt.title('Correlation of ddG/dSASA with sc_value')
        plt.axvline(x=sc_value_quartile, color='r', linestyle='--')
        plt.text(sc_value_quartile, dG_dSASA_quartile - 0.1, 'best quartile sc_value', rotation=90)
        plt.axhline(y=dG_dSASA_quartile, color='r', linestyle='--')
        plt.text(sc_value_quartile - 0.4, dG_dSASA_quartile + 0.1, 'best quartile ddG/dSASA')
        plt.savefig(os.path.join(self.plots_folder, 'ddG_dSASA_vs_sc_value.png'))

    def integrate_geometry_scores(self, df_rosetta):
        """Step 6: Integrate geometry scores with Rosetta scores."""
        geometries_scores_df = pd.read_csv(self.geometries_scores_file)
        
        not_found = []

        for i, row in df_rosetta.iterrows():
            geometry_row = geometries_scores_df[geometries_scores_df['pdb_name'].str.replace('.pdb', '') == row['geometry']]

            if not geometry_row.empty:
                df_rosetta.at[i, 'total_geometry_score'] = geometry_row['total_geometry_score'].values[0]
                df_rosetta.at[i, 'geometry_score'] = geometry_row['geometry_score'].values[0]
                df_rosetta.at[i, 'n_contacts_geometry'] = geometry_row['n_contacts'].values[0]
                df_rosetta.at[i, 'cluster_label'] = geometry_row['cluster_label'].values[0]
            else:
                if row['geometry'] not in not_found:
                    not_found.append(row['geometry'])
                df_rosetta.at[i, 'total_geometry_score'] = np.nan
                df_rosetta.at[i, 'geometry_score'] = np.nan
                df_rosetta.at[i, 'n_contacts_geometry'] = np.nan
                df_rosetta.at[i, 'cluster_label'] = np.nan

        if not_found:
            print(f"WARNING: The following geometries were not found in the geometry scores file: {not_found}")

        df_rosetta.to_csv(f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/OptimizedSequenceswithGeometryScores.csv', index=False)

    def generate_per_cluster_plots(self, df_rosetta):
        """Step 7: Generate cluster plots for scoring metrics."""
        print("Generating cluster plots...")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster_label', y='dG_separated', data=df_rosetta)
        plt.xlabel('Cluster label')
        plt.ylabel('dG_separated')
        plt.title('Distribution of ddG per cluster')
        plt.savefig(os.path.join(self.plots_folder, 'ddG_per_cluster.png'))

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster_label', y='dG_dSASA', data=df_rosetta)
        plt.xlabel('Cluster label')
        plt.ylabel('ddG/dSASA')
        plt.title('Distribution of ddG/dSASA per cluster')
        plt.savefig(os.path.join(self.plots_folder, 'ddG_dSASA_per_cluster.png'))

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster_label', y='sc_value', data=df_rosetta)
        plt.xlabel('Cluster label')
        plt.ylabel('Shape Complementarity value')
        plt.title('Shape Complementarity per cluster')
        plt.savefig(os.path.join(self.plots_folder, 'sc_value_per_cluster.png'))

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster_label', y='polar_percentage', data=df_rosetta)
        plt.xlabel('Cluster label')
        plt.ylabel('Polar fraction')
        plt.title('Polar fraction per cluster')
        plt.savefig(os.path.join(self.plots_folder, 'polar_fraction_per_cluster.png'))

        print(f"Cluster plots saved to {self.plots_folder}.")
        

    def select_best_designs(self, df_rosetta):
        """Step 8: Select the top designs based on ddG, ddG/dSASA, and shape complementarity (sc_value)."""
        
        print("Selecting top designs based on best ddG, ddG/dSASA, sc_value, and low alanine content...")
        
        # Define the quartiles for the selection criteria
        sc_value_quartile = df_rosetta['sc_value'].quantile(0.75)
        dG_dSASA_quartile = df_rosetta['dG_dSASA'].quantile(0.25)
        dG_separated_quartile = df_rosetta['dG_separated'].quantile(0.25)
        
        # Filter the dataframe based on the best quartiles and alanine content
        df_best = df_rosetta[
            (df_rosetta['dG_separated'] < dG_separated_quartile) &
            (df_rosetta['dG_dSASA'] < dG_dSASA_quartile) &
            (df_rosetta['sc_value'] > sc_value_quartile) &
            (df_rosetta['ala_count'] < 4)
        ]
        
        print(f"Number of designs in the best quartiles: {df_best.shape[0]}")

        # Select the top 10 designs from each cluster
        df_top = pd.DataFrame()
        for cluster in df_best['cluster_label'].unique():
            df_cluster = df_best[df_best['cluster_label'] == cluster]
            df_cluster = df_cluster.sort_values(by='dG_separated', ascending=True)
            df_top = pd.concat([df_top, df_cluster.head(10)], axis=0)

        if df_top.empty:
            df_top = df_rosetta.copy()

        output_top_designs = f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/TopRosettaScoreDesigns.csv'
        df_top.to_csv(output_top_designs, index=False)
        print(f"Top designs saved to {output_top_designs}")
        print(f"Number of selected top designs: {df_top.shape[0]}")

        # Copy the selected PDB files to the SelectedDesigns folder
        selected_designs_folder = f'{self.mpnn_output_base_path}/{self.nsym}mer/{self.nsym}_r{self.radius}/SelectedDesigns'
        os.makedirs(selected_designs_folder, exist_ok=True)

        for pdb_path in df_top['description']:
            os.system(f'cp {pdb_path} {selected_designs_folder}')

        # Rename the PDB files to remove '_0001' suffix
        for filename in os.listdir(selected_designs_folder):
            if filename.endswith('_0001.pdb'):
                new_filename = filename.replace('_0001.pdb', '.pdb')
                old_file = os.path.join(selected_designs_folder, filename)
                new_file = os.path.join(selected_designs_folder, new_filename)
                os.rename(old_file, new_file)
        
        print(f"Selected PDB files copied and renamed in {selected_designs_folder}.")


    def run_pipeline(self):
        """Runs the entire Rosetta scoring pipeline."""
        self.create_rosetta_jobfile()
        print("Submitting Rosetta FastRelax and InterfaceAnalyzer jobfile for scoring...")
        self.submit_rosetta_jobfile()
        df_rosetta = self.concatenate_rosetta_scores()
        print("Analyzing Rosetta scores and creating plots...")
        df_rosetta = self.analyze_rosetta_scores()
        self.integrate_geometry_scores(df_rosetta)
        print("Generating per-cluster plots...")
        self.generate_per_cluster_plots(df_rosetta)
        print("Selecting best designs...")
        self.select_best_designs(df_rosetta)
        print("Rosetta scoring pipeline complete.")
