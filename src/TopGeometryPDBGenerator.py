import pandas as pd
import argparse
from scipy.spatial.transform import Rotation as R
import numpy as np
from pyrosetta import *
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.core.pose import Pose
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB import MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
import os
init()

parser = argparse.ArgumentParser(description="Script to generate protein rings based on monomer structures.")
parser.add_argument('--monomer_pdb_path', type=str, required=True, help='Path to the monomer PDB file, e.g., /path/to/monomer.pdb')
parser.add_argument('--geometry_folder_path', type=str, required=True, help='Path to the folder containing geometry and scores, e.g., /path/to/folder/Outputs-GeometryAnalizer/MonomerName/Fixed_nSym_radius')
parser.add_argument('--output_folder', type=str, required=True, help='Path to output folder for saving results, e.g., /path/to/output')
parser.add_argument('--scores_folder_path', type=str, required=True, help='Path to the folder containing scores, e.g., /path/to/folder/Scores')
parser.add_argument('--N', type=int, default=None, help='Number of conformations to generate (default: all available)')
parser.add_argument('--whole_ring', action='store_true', help='Generate the whole ring')
args = parser.parse_args()

MONOMER_NAME = os.path.basename(args.monomer_pdb_path).replace('.pdb', '')
MINIMUM_SCORE = 0.4

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
    
if not os.path.exists(args.scores_folder_path):
    os.makedirs(args.scores_folder_path)

geometry_scores_file = os.path.join(args.geometry_folder_path, 'geometry_total_scores.csv')
df = pd.read_csv(geometry_scores_file)
df = df.drop_duplicates()
df = df.sort_values(by=['score*n_contacts'], ascending=False)
df = df.reset_index(drop=True)
print("******* Geometry file reading process completed.")
print("******* Total number of poses: {}".format(len(df)))

if args.N is None:
    N = len(df)
else:
    N = args.N
print("******* Number of conformations to generate: {}".format(N))

df['score'] = df['score'].astype(float)
df['n_contacts'] = df['n_contacts'].astype(int)
df['score*n_contacts'] = df['score*n_contacts'].astype(float)
df = df[df['score'] > MINIMUM_SCORE]
print("******* Total number of poses with score higher than 0.4: {}".format(len(df)))
df = df.reset_index(drop=True)

# 4. Save the name of the generated pdb and the scores read from the geometry file into a csv file in the Scores folder
df_names = pd.DataFrame(columns=['pdb_name', 'nSym', 'radius', 'rotx', 'roty', 'rotz', 'total_geometry_score', 'geometry_score', 'n_contacts'])
rows_list = []
for index, row in df.head(N).iterrows():
    pdb_name = f'{MONOMER_NAME}_{int(row["nSym"])}_r{int(row["radius"])}_rot{round(row["rotx"], 2)}_{round(row["roty"], 2)}_{round(row["rotz"], 2)}_score{round(row["score"], 3)}.pdb'
    new_row = {'pdb_name': pdb_name, 
               'nSym': row['nSym'], 
               'radius': row['radius'], 
               'rotx': row['rotx'], 
               'roty': row['roty'], 
               'rotz': row['rotz'], 
               'total_geometry_score': row['score*n_contacts'], 
               'geometry_score': row['score'], 
               'n_contacts': row['n_contacts']}
    rows_list.append(new_row)
df_names = pd.concat([df_names, pd.DataFrame(rows_list)], ignore_index=True)

score_output_path = os.path.join(args.scores_folder_path, f'{os.path.basename(args.geometry_folder_path)}.csv')
df_names.to_csv(score_output_path, index=False)
print("Score saving complete.")

# 3. Generate the conformations with pyRosetta
def apply_rotation(pose, rotation_matrix):
    """Apply rotation to the given pose using rotation matrix and set_xyz."""
    for i in range(1, pose.total_residue() + 1):
        for j in range(1, pose.residue(i).natoms() + 1):
            coord = pose.residue(i).xyz(j)
            rotated_coord = rotation_matrix @ np.array([coord.x, coord.y, coord.z])
            new_coord = xyzVector_double_t(rotated_coord[0], rotated_coord[1], rotated_coord[2])
            pose.residue(i).set_xyz(j, new_coord)

def transform_points_pyrosetta(base_pose, radius, rotx, roty, rotz, nSym, whole_ring=False):
    rot1 = R.from_euler('x', 90, degrees=True)
    rot2 = R.from_euler('xyz', [rotx, roty, rotz], degrees=True)
    rotation_matrix1 = rot1.as_matrix()
    rotation_matrix2 = rot2.as_matrix()

    apply_rotation(base_pose, rotation_matrix1)
    apply_rotation(base_pose, rotation_matrix2)

    transformed_poses = []
    x = 2
    if whole_ring:
        x = nSym

    for index in range(x):
        pose = Pose()
        pose.assign(base_pose)

        angle = 360 / nSym * index
        rot3 = R.from_euler('z', angle, degrees=True)
        rotation_matrix3 = rot3.as_matrix()
        apply_rotation(pose, rotation_matrix3)

        # Translation mover
        translation = np.array([
            radius * np.cos(np.deg2rad(angle)),
            radius * np.sin(np.deg2rad(angle)),
            0
        ])
        # Convert the vector to Rosetta's xyzVector_double_t
        translation_vector = xyzVector_double_t(translation[0], translation[1], translation[2])
        
        # Loop through all residues and all atoms to apply the translation
        for i in range(1, pose.total_residue() + 1):
            for j in range(1, pose.residue(i).natoms() + 1):
                pose.residue(i).set_xyz(j, pose.residue(i).xyz(j) + translation_vector)
        transformed_poses.append(pose)
    return transformed_poses

def merge_pdb_ring(file_list, output_file):
    parser = PDBParser()
    structures = []
    for file in file_list:
        structure = parser.get_structure("structure", file)
        structures.append(structure)
    merged_structure = structures[0].copy()
    for structure in structures[1:]:
        for chain in structure.get_chains():
            chain.id = chr(ord(list(merged_structure.get_chains())[-1].id) + 1)
            chain.detach_parent()
            merged_structure[0].add(chain)
    mmcifio = MMCIFIO()
    mmcifio.set_structure(merged_structure)
    mmcifio.save(output_file)

def merge_pdb(file1, file2, output_file):
    parser = PDBParser()
    structure1 = parser.get_structure("structure1", file1)
    structure2 = parser.get_structure("structure2", file2)

    chains = list(structure2.get_chains())
    chains[0].id = 'C'
    chains[1].id = 'D'
    chains[0].detach_parent()
    chains[1].detach_parent()
    structure1[0].add(chains[0])
    structure1[0].add(chains[1])

    io = PDBIO()
    io.set_structure(structure1)
    io.save(output_file)
print(f"******* Generating {N} PDBs from top-scoring poses.")


## corrected
for i in range(N): 
    nSym, radius, corrected_radius, rotx, roty, rotz = int(df.iloc[i]['nSym']), int(df.iloc[i]['radius']), float(df.iloc[i]['corrected_radius']), float(df.iloc[i]['rotx']), float(df.iloc[i]['roty']), float(df.iloc[i]['rotz'])
    score = round(df.iloc[i]['score'], 3)
    output_folder = os.path.join(args.output_folder, f"{MONOMER_NAME}_{nSym}_r{radius}_rot{round(rotx, 2)}_{round(roty, 2)}_{round(rotz, 2)}_score{score}")

    if args.whole_ring:
        output_file = f"{output_folder}_wholering.cif"
        if os.path.exists(output_file):
            print(f"PDB {i + 1} already exists. Skipping...")
            continue
        
        base_pose = pose_from_pdb(args.monomer_pdb_path)   
        print(f"Generating PDB {i + 1}...")
        print(f"nSym: {nSym}, corrected_radius: {corrected_radius}, rotx: {rotx}, roty: {roty}, rotz: {rotz}")
        
        transformed_poses = transform_points_pyrosetta(base_pose, corrected_radius, rotx, roty, rotz, nSym, whole_ring=True)
        pdb_files_to_merge = []
        
        for j in range(nSym):
            temp_pdb_path = f"{output_folder}_TEMP_{j + 1}.pdb"
            transformed_poses[j].dump_pdb(temp_pdb_path)
            pdb_files_to_merge.append(temp_pdb_path)

        merge_pdb_ring(pdb_files_to_merge, output_file)
        
        # Optionally remove temporary files
        # for temp_pdb in pdb_files_to_merge:
        #     os.remove(temp_pdb)
        del pdb_files_to_merge, output_merged_file
          
    else:
        output_file = f"{output_folder}.pdb"
        if os.path.exists(output_file):
            print(f"PDB {i + 1} already exists. Skipping...")
            continue
        
        base_pose = pose_from_pdb(args.monomer_pdb_path)
        print(f"Generating PDB {i + 1}...")
        print(f"nSym: {nSym}, corrected_radius: {corrected_radius}, rotx: {rotx}, roty: {roty}, rotz: {rotz}")
        
        transformed_poses = transform_points_pyrosetta(base_pose, corrected_radius, rotx, roty, rotz, nSym)
        temp_pdb_path_1 = f"{output_folder}_TEMP_1.pdb"
        temp_pdb_path_2 = f"{output_folder}_TEMP_2.pdb"
        transformed_poses[0].dump_pdb(temp_pdb_path_1)
        transformed_poses[1].dump_pdb(temp_pdb_path_2)
        merge_pdb(temp_pdb_path_1, temp_pdb_path_2, output_file)
        
        # Remove temporary files
        os.remove(temp_pdb_path_1)
        os.remove(temp_pdb_path_2)
    del transformed_poses, nSym, radius, corrected_radius, rotx, roty, rotz, score, base_pose

print ("PDB generation complete.")