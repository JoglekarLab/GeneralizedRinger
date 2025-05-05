#!/bin/bash
#SBATCH --job-name=16200PDB
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=4:00:00
#SBATCH --account=ajitj99

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /home/linamp/miniconda3/envs/myenv_gl/envs/rosetta_env

# Generate dimer PDB files
python TopGeometryPDBGenerator.py --monomer_pdb_path ../0_Inputs/N271_xtal.pdb --geometry_folder_path ../1_Geometry_Selection/16mer/16_r200/Outputs-GeometryAnalizer/N271_xtal/Fixed_nSym16_radius200 --output_folder ../1_Geometry_Selection/16mer/16_r200/Outputs-GeometryAnalizer/N271_xtal/Fixed_nSym16_radius200/2-Dimers --scores_folder_path ../1_Geometry_Selection/16mer/16_r200/Outputs-GeometryAnalizer/N271_xtal/Scores/N271_xtal --N 500
