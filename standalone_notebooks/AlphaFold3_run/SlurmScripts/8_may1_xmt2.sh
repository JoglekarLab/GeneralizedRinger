#!/bin/bash
#SBATCH --job-name=AF3
#SBATCH --output=SlurmScripts/AF3_%j.out
#SBATCH --error=SlurmScripts/AF3_%j.err
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:40:00
#SBATCH --mem=128GB
#SBATCH --account=ajitj99

echo "CURRENT WORKING DIRECTORY = $(pwd)"

# Optimize GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # Disables memory preallocation
export TF_FORCE_UNIFIED_MEMORY=true             # Enables unified memory (if GPU RAM fills up it can spill into host CPU RAM)
export XLA_CLIENT_MEM_FRACTION=3.2              # Your job can request up to 3.2 times the GPU memory due to the unified memory setting
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_disable_hlo_passes=custom-kernel-fusion-rewriter" # There are known numerical issues with CUDA Capability 7.x devices. To work around the issue, set the ENV XLA_FLAGS to include --xla_disable_hlo_passes=custom-kernel-fusion-rewriter

# Load AF3
# Note that this version of alphafold is run in a Python (3.11) "venv" virtual environment. 
# The required cuda libraries are packaged with this version of alphafold so there is no need to load the cuda module to run alphafold on GPUs
module load Bioinformatics alphafold/3.0.0

# Run AF3
# example: python $AF3_SCRIPT_DIR/run_alphafold.py --json_path=$AF3_PARAMS_DIR/input/fold_input.json --output_dir=your_output_dir --db_dir=$AF3_PARAMS_DIR/db_dir --model_dir=$AF3_PARAMS_DIR/af3_model_params
python $AF3_SCRIPT_DIR/run_alphafold.py --json_path=JsonInputs/8.json --output_dir=AF3_outputs/8 --db_dir=$AF3_PARAMS_DIR/db_dir --model_dir=$AF3_PARAMS_DIR/af3_model_params

module unload alphafold
