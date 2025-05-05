import os
import subprocess
import time
import pandas as pd
from geometry_functions import CA_coords, get_score, get_all_pairs, general_get_contacts
import matplotlib.pyplot as plt

class GeometryScoringPipeline:
    def __init__(self, sym, radius, pdb_inputs_dir, output_dir, omit_chain_pairs=[]):
        
        self.sym = sym
        self.radius = radius
        self.omit_chain_pairs = omit_chain_pairs
        
        #Paths
        self.pdb_inputs_dir = pdb_inputs_dir
        self.pdb_files = [f for f in os.listdir(self.pdb_inputs_dir) if f.endswith('.pdb')]
        self.pdb_files = [os.path.join(self.pdb_inputs_dir, f) for f in self.pdb_files]
        self.output_dir = os.path.join(output_dir, f"{self.sym}mer", f"{self.sym}_r{self.radius}")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def scorer (self, pdb_file):
        pairs = get_all_pairs(pdb_file, self.omit_chain_pairs)
        distances, contacts_index, n_contacts = general_get_contacts(pairs)
        score = get_score(distances)
        return score, distances, contacts_index, n_contacts
        
    def get_scores_info_pdb(self, pdb_file, cluster_label=0, rotx=0, roty=0, rotz=0):
        """Calculates the scores for a given PDB file and gives the info to create a row in the csv file"""
        "should be like pdb_name,nSym,radius,rotx,roty,rotz,total_geometry_score,geometry_score,n_contacts,cluster_label" 
        "N271_xtal_12_r150_rot46.21_299.47_148.56_score0.639.pdb,12.0,150.0,46.21124267578125,299.46807861328125,148.55987548828125,19.811485848607752,0.6390801886647662,31.0,0"
        score, distances, contacts_index, n_contacts = self.scorer(pdb_file)
        filename = os.path.basename(pdb_file)
        row = [
            filename,
            self.sym,
            self.radius,
            rotx,
            roty,
            rotz,
            score,
            score / n_contacts if n_contacts > 0 else 0,
            n_contacts,
            cluster_label
        ]
        return row
    

    def run_pipeline(self):
        output_file = os.path.join(self.output_dir, f"SelectedGeometries_Scores.csv")
        header = ["pdb_name", "nSym", "radius", "rotx", "roty", "rotz", "total_geometry_score", "geometry_score", "n_contacts", "cluster_label"]
        
        # Stream-write not to hold everything in memory
        with open(output_file, 'w') as f:
            f.write(','.join(header) + '\n')
            for pdb_file in self.pdb_files:
                row = self.get_scores_info_pdb(pdb_file)
                f.write(','.join(map(str, row)) + '\n') # map(str, row) applies str() to each element of row, to be able to join
                
        print(f"Scores for {len(self.pdb_files)} PDB files have been written to {output_file}")
                    
                    
        
                    
            