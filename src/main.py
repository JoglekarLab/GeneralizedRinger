from alphafold3_scoring_pipeline import AlphaFold_3_ScoringPipeline
from design_generator_pipeline import DesignGeneratorPipeline
from geometry_optimization_pipeline import GeometryOptimizationPipeline
from rosetta_scoring_pipeline import RosettaScoringPipeline
from geometry_scoring_pipeline import GeometryScoringPipeline
from job_manager import JobManager
import multiprocessing
import argparse
import os
import itertools

parser = argparse.ArgumentParser(description="Run design pipelines with specified parameters")
parser.add_argument('--radius', type=int, required=True, help="Radius value")
parser.add_argument('--nsym', type=int, required=True, help="Symmetry number")
args = parser.parse_args()

nsym = args.nsym
radius = args.radius
monomer_pdb_path = "../0_Inputs/N271_xtal.pdb"

pyrosetta_env = "/nfs/turbo/umms-ajitj/conda_envs/rosetta_env"
general_env = "/nfs/turbo/umms-ajitj/conda_envs/myenv"
account = "ajitj99"
rosetta_exec_path = "$ROSETTA3/bin/rosetta_scripts.default.linuxgccrelease"  # Path to Rosetta executable
FastRelax_protocol_path = 'FastRelax_InterfaceAnalysis.xml' # Path to FastRelax plus InterfaceAnalyzer script

print(f"**** Running pipeline for {nsym}mer with diameter {radius*2/10} Angstroms.\nMonomer PDB path: {monomer_pdb_path} ****")

# Additional Functions
def run_pipeline_function(pipeline, *args):
    """
    Runs a given pipeline with provided arguments.
    
    The *args allows for passing a variable number of arguments to the function.
    These arguments are then passed to the pipeline's run_pipeline method.
    Essentially, *args collects all additional positional arguments and makes 
    them accessible as a tuple.
    """
    pipeline.run_pipeline(*args)


# PARAMETERS
# Parameters for selecting design chains (DesignGeneratorPipeline):
# max_distance = 8.5
# cb_distance = 7
max_distance = 100
cb_distance = 7
extended_design = True
interface = "A_BC"
fixed_chains = "B,C"

# Parameters for ProteinMPNN.
# Importantly, note that you can select a different dictionary for the second round.
# temperature_to_seq_count = {
#     0.15: 35,
#     0.25: 95,
#     0.30: 120,
#     0.35: 120
# }

# Number of sequences to generate per temperature
temperature_to_seq_count = {
    0.15: 10,
    0.25: 10,
    0.30: 10,
    0.35: 10
}

# AF3
seeds_number=1
predefined_seeds=False
predefined_seeds_list=None
dialect="alphafold3" # Usually, stick to this
version=1 # Usually, stick to this
pae_omit_pairs = [['B','C']]
plddt_chains = ['A']

msas_path= "../0_Inputs/msa_inputs/7b1f_data.json"
msa_chains=['B','C'] # e.g. ['B','C']. Will determine which chains to use the unpairedMsas in msas_path for.
template_chains=False # e.g. ['B','C']. Will determine which chains to use the templates in msas_path for.

dirs = [
    "0_Inputs",
    "1_Geometry_Selection",
    "2_Geometry_Clustering",
    "3_MPNN_Design",
    "4_Alphafold_Predictions",
    "6_GenerateDesigns",
    "7_Alphafold_Predictions",
    "8_Final_Selection",
]

for d in dirs:
    os.makedirs(os.path.join("..", d), exist_ok=True)

msa_folder = os.path.join("..", "0_Inputs", "msa_inputs")
os.makedirs(msa_folder, exist_ok=True)

# GENERAL 
# Paths
clustering_base_path = "../2_Geometry_Clustering"
source_folder_geometries = f'{clustering_base_path}/{nsym}mer/{nsym}_r{radius}/SelectedGeometries'
mpnn_output_base_path = "../3_MPNN_Design"

# !!!! Copy the geometries you want to analyze to the source_folder_geometries folder !!!!
os.makedirs(source_folder_geometries, exist_ok=True)
pdb_files = [f for f in os.listdir(source_folder_geometries) if f.lower().endswith('.pdb')]
assert pdb_files, (
    f"No .pdb files found in {source_folder_geometries}.\n"
    "Remember to copy the geometries you want to analyze to {source_folder_geometries}"
)

# Check that all the files in source_folder_geometries are in lowercase. If there are any capital letters put it in lowercase
for fname in pdb_files:
    original = os.path.join(source_folder_geometries, fname)
    lower_name = fname.lower()
    lower_path = os.path.join(source_folder_geometries, lower_name)
    if fname != lower_name:
        os.rename(original, lower_path)
        print(f"Renamed {fname}")

# FIRST ROUND
# Geometry Pipeline
# methods = ["annealing", "shgo", "brute"]
# GOpipeline = GeometryOptimizationPipeline(nsym, radius, monomer_pdb_path, methods, pyrosetta_env, general_env, account)
# GOpipeline.run_pipeline()

# MPNN Design Pipeline
source_folder_geometries = f'{clustering_base_path}/{nsym}mer/{nsym}_r{radius}/SelectedGeometries'
DGpipeline = DesignGeneratorPipeline(nsym, radius, pyrosetta_env, general_env, account, temperature_to_seq_count, mpnn_output_base_path, source_folder_geometries, interface, fixed_chains, cb_distance, extended_design, max_distance)
DGpipeline.run_pipeline()

# Generate geometry scores (these will be generated in the clustering folder. Why? Because that was how the original pipeline was made)
# If you want everything to work downstream, just keep the output in the 2_Geometry_Clustering folder
# Inputs dir is the source_folder_geometries
# Outputs dir is the clustering_base_path with the selected symmetry and radius

# GeometryScoringPipeline
fixed = fixed_chains.split(",")
chain_pairs_to_omit = list(itertools.combinations(fixed, 2))
print(f"Chain pairs to omit: {chain_pairs_to_omit}")
GSpipeline = GeometryScoringPipeline(nsym, radius, source_folder_geometries, clustering_base_path, omit_chain_pairs=chain_pairs_to_omit)
GSpipeline.run_pipeline()

# Rosetta scoring pipeline
mpnn_output_base_path = "../3_MPNN_Design"
RSpipeline=RosettaScoringPipeline(nsym, radius, pyrosetta_env, account, mpnn_output_base_path, clustering_base_path, FastRelax_protocol_path, rosetta_exec_path)
RSpipeline.run_pipeline()

# Alphafold pipeline
pdb_input_folder_path = f'../3_MPNN_Design/{nsym}mer/{nsym}_r{radius}/SelectedDesigns'
AF_folder = '../4_Alphafold_Predictions/'
rosetta_file_path = f'../3_MPNN_Design/{nsym}mer/{nsym}_r{radius}/RelaxedOptimizationScores.csv'
designs_scores = f'../3_MPNN_Design/{nsym}mer/{nsym}_r{radius}/TopRosettaScoreDesigns.csv'
AFpipeline = AlphaFold_3_ScoringPipeline(nsym, radius, account, general_env, pdb_input_folder_path, AF_folder, seeds_number, predefined_seeds, predefined_seeds_list, dialect, version, pae_omit_pairs, plddt_chains)
AFpipeline.run_pipeline(rosetta_file_path, designs_scores)

# SECOND ROUND
# Second round of Pipeline to generate designs using ProteinMPNN
mpnn_output_base_path = "../6_GenerateDesigns"
source_folder_geometries_path = f'../3_MPNN_Design/{nsym}mer/{nsym}_r{radius}/SelectedDesigns'

DGpipeline = DesignGeneratorPipeline(nsym, radius, pyrosetta_env, general_env, account, temperature_to_seq_count, mpnn_output_base_path, source_folder_geometries_path, interface, fixed_chains, cb_distance, extended_design, max_distance)
DGpipeline.run_pipeline()

# Second round of Rosetta scoring pipeline
mpnn_output_base_path = "../6_GenerateDesigns"
clustering_base_path = "../2_Geometry_Clustering"
RSpipeline=RosettaScoringPipeline(nsym, radius, pyrosetta_env, account, mpnn_output_base_path, clustering_base_path, FastRelax_protocol_path, rosetta_exec_path)
RSpipeline.run_pipeline()

# Second round of Alphafold pipeline
pdb_input_folder_path = f'../6_GenerateDesigns/{nsym}mer/{nsym}_r{radius}/SelectedDesigns'
AF_folder = '../7_Alphafold_Predictions/'
rosetta_file_path = f'../6_GenerateDesigns/{nsym}mer/{nsym}_r{radius}/RelaxedOptimizationScores.csv'
designs_scores = f'../6_GenerateDesigns/{nsym}mer/{nsym}_r{radius}/TopRosettaScoreDesigns.csv'
AFpipeline = AlphaFold_3_ScoringPipeline(nsym, radius, account, general_env, pdb_input_folder_path, AF_folder, seeds_number, predefined_seeds, predefined_seeds_list, dialect, version, pae_omit_pairs, plddt_chains)
AFpipeline.run_pipeline(rosetta_file_path, designs_scores)

print(f"Congrats! Your design pipeline for radius {radius} and symmetry number of {nsym} for the monomer {monomer_pdb_path} has finished!")

# err_out_files = ['*.err', '*.out']
# err_out_folder = f'../{nsym}mer_{radius}_err_out_files'
# os.system(f'mkdir -p {err_out_folder}')
# for file in err_out_files:
#     os.system(f'mv {file} {err_out_folder}')
