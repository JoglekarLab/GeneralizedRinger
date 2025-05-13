#!/bin/bash
#SBATCH --job-name=ProtMPNN_%j
#SBATCH --output=ProtMPNN_%j.out
#SBATCH --error=ProtMPNN_%j.err
#SBATCH --partition=gpu,gpu_mig40,spgpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00
#SBATCH --account=ajitj99

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /nfs/turbo/umms-ajitj/conda_envs/protmpnn_env

INPUT_DIR="$1"  # Input directory e.g., ../Outputs-GeometryAnalizer/
OUTPUT_DIR="$2"  # Output directory e.g., ../Outputs-ProteinMPNN
NUM_SEQ_PER_TARGET=$3
SAMPLING_TEMP="$4"

# Create the output directory if it does not exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Set up a temporary directory based on INPUT_DIR
TEMP_DIR="${INPUT_DIR}/TEMP"
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi
mkdir -p "$TEMP_DIR"

PROTEIN_MPNN_PATH="/nfs/turbo/umms-ajitj/ProteinMPNN"

for pdb_file in "${INPUT_DIR}"/*.pdb; do
    printf "$pdb_file"
    PDB_NAME=$(basename "$pdb_file" .pdb)

    # Copy pdb and its corresponding design file to TEMP_DIR
    cp "$pdb_file" "$TEMP_DIR/"
    cp "${INPUT_DIR}/${PDB_NAME}_design_residues.txt" "$TEMP_DIR/"

    input_design_residues="$TEMP_DIR/${PDB_NAME}_design_residues.txt"
    input_folder="$TEMP_DIR"

    chains_to_design=$(awk 'NR==1' $input_design_residues)
    design_only_positions=$(awk 'NR==2' $input_design_residues)

    printf "Input pdb: %s\n" "$PDB_NAME"
    printf "Chains to design: %s\n" "$chains_to_design"
    printf "Design only positions: %s\n" "$design_only_positions"

    output_dir="${OUTPUT_DIR}/${PDB_NAME}"
    mkdir -p $output_dir
    printf "Output dir: %s\n" "$output_dir"

    path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
    path_for_assigned_chains=$output_dir"/assigned_pdbs.jsonl"
    path_for_fixed_positions=$output_dir"/fixed_pdbs.jsonl"
    path_for_bias=$PROTEIN_MPNN_PATH/helper_scripts/antielectrostatic_bias_pdbs.jsonl

    python $PROTEIN_MPNN_PATH/helper_scripts/parse_multiple_chains.py --input_path=$input_folder --output_path=$path_for_parsed_chains
    python $PROTEIN_MPNN_PATH/helper_scripts/assign_fixed_chains.py --input_path=$path_for_parsed_chains --output_path=$path_for_assigned_chains --chain_list "$chains_to_design"
    python $PROTEIN_MPNN_PATH/helper_scripts/make_fixed_positions_dict.py --input_path=$path_for_parsed_chains --output_path=$path_for_fixed_positions --chain_list "$chains_to_design" --position_list "$design_only_positions" --specify_non_fixed
    python $PROTEIN_MPNN_PATH/protein_mpnn_run.py \
            --jsonl_path $path_for_parsed_chains \
            --chain_id_jsonl $path_for_assigned_chains \
            --fixed_positions_jsonl $path_for_fixed_positions \
            --bias_AA_jsonl $path_for_bias \
            --out_folder $output_dir \
            --num_seq_per_target "$NUM_SEQ_PER_TARGET" \
            --batch_size 1 \
            --sampling_temp "$SAMPLING_TEMP"
    
    Remove the temporary files in $TEMP_DIR
    rm -rf "$TEMP_DIR"/*
done

Optionally, remove the TEMP_DIR entirely if no longer needed
rm -rf "$TEMP_DIR"

# Copy the fasta files of the designed sequences to the final output directory
mkdir -p "${OUTPUT_DIR}/T_${SAMPLING_TEMP}"
cp "${OUTPUT_DIR}"/*/"seqs"/* "${OUTPUT_DIR}/T_${SAMPLING_TEMP}/"

# Remove intermediate directories unless they start with T_
for dir in "${OUTPUT_DIR}"/*; do
    if [[ "$dir" != "${OUTPUT_DIR}/T_"* ]]; then
        rm -rf "$dir"
    fi
done

