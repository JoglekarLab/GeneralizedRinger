from Bio import SeqIO
import os
import argparse
import pandas as pd

# Usage: python Extract_TopMPNN.py --folder ../Outputs-ProteinMPNN/A271/Fixed_nSym16_radius155/ --N 2

parser = argparse.ArgumentParser(description="Process fasta files.")
parser.add_argument('--scores_file', required=False, default=None, help='The scores file.')
parser.add_argument('--mpnn_folder', required=True, help='The folder containing the MPNN generated results.fa files.')
parser.add_argument('--N', type=int, required=False, default=None, help='The number of top sequences to extract.')
args = parser.parse_args()

# Score file paths
# [MONOMER_NAME,FOLDER_NAME] = args.folder.split('/')[-2:]
# score_df = pd.read_csv(f"../Scores/{MONOMER_NAME}/{FOLDER_NAME}.csv")
if args.scores_file is not None:
    score_df = pd.read_csv(args.scores_file)
    new_score_df = pd.DataFrame(columns=['pdb_name', 'nSym', 'radius', 'rotx', 'roty', 'rotz', 'total_geometry_score', 'geometry_score', 'n_contacts','proteinMPNN_score', 'proteinMPNN_global_score'])

def process_file(input_file, output_dir, N=args.N):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seqs = list(SeqIO.parse(input_file, "fasta"))
    sort_seqs = sorted(seqs, key=lambda x: float(x.description.split('global_score=')[1].split(',')[0])) #ascending order
    # drop duplicates
    seen = set()
    sort_seqs = [x for x in sort_seqs if not (x.seq in seen or seen.add(x.seq))]
    
    if N is None:
        N = len(sort_seqs)
    for i in range(N):
        if args.scores_file is not None:
            # Write scores
            complete_name = os.path.basename(input_file).replace(".fa", ".pdb")
            name = complete_name.split('_')[1:]
            name = '_'.join(name)
            print(name)
            row = score_df[score_df.pdb_name==name].values[0].tolist()
            row.append(float(sort_seqs[i].description.split(' score=')[1].split(',')[0]))
            row.append(float(sort_seqs[i].description.split('global_score=')[1].split(',')[0]))
            row[0] = os.path.basename(input_file).replace(".fa", "_rank" + str(i+1))
            new_score_df.loc[len(new_score_df)] = row
        # Write fasta
        output_file = os.path.join(output_dir, os.path.basename(input_file).replace(".fa", "") + '_rank' + str(i+1) + '.fa')  
        sort_seqs[i].seq = sort_seqs[i].seq.replace('/', ':') # To use as input for ColabFold
        sort_seqs[i].id = os.path.basename(input_file).split('.')[0] + '_rank' + str(i+1)
        # sort_seqs[i].description = sort_seqs[i].description.split('global_score=')[0]
        SeqIO.write(sort_seqs[i], output_file, "fasta")
        print(f'Wrote {os.path.basename(output_file)}...')

for file_name in os.listdir(args.mpnn_folder):
    if file_name.endswith('.fa'):
        input_file = os.path.join(args.mpnn_folder, file_name)
        output_dir = os.path.join(args.mpnn_folder, "TopSequences")
        if not os.path.exists(output_dir):
            print(f'Creating {output_dir}...')
            os.makedirs(output_dir)
        process_file(input_file, output_dir)

if args.scores_file is not None:
    new_score_df.to_csv(args.scores_file.replace(".csv", "_top" + str(args.N) + ".csv"), index=False)