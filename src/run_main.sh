#!/bin/bash
#SBATCH --job-name=16-200
#SBATCH --output=main_%j.out
#SBATCH --error=main_%j.err
#SBATCH --partition=standard
#SBATCH --mem=4G
#SBATCH --time=70:00:00
#SBATCH --account=ajitj99

radius=0
nsym=0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /nfs/turbo/umms-ajitj/conda_envs/myenv

# Run the Python script and pass the radius and nsym as arguments
python main.py --radius "$radius" --nsym "$nsym"
