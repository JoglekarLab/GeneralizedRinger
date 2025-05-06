'''
Usage: python geometry_clustering.py --input_geometries_folder /path/to/geometries/folder --output_cluster_folder /path/to/output_folder --scores_file /path/to/scores/file
'''

import os
import numpy as np
import random
import argparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

import os
import numpy as np
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import umap


def read_pdb(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb_structure', file_path)
    coordinates = []

    for model in structure:
        for chain in model:
            for atom in chain.get_atoms():
                if atom.get_name() == 'CA':
                    coords = atom.get_coord()
                    coordinates.append(coords)
    # Flatten the coordinates list
    return np.array(coordinates).flatten()

def get_data_matrix(pdb_folder, min_max=True):
    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith('.pdb')]
    data_matrix = None
    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_folder, pdb_file)
        pdb_data = read_pdb(pdb_path)
        if data_matrix is None:
            data_matrix = pdb_data
        else:
            data_matrix = np.vstack((data_matrix, pdb_data))
    if min_max: # Rescale the data matrix to [0,1]
        data_matrix_zero_center = data_matrix - np.mean(data_matrix, axis=0)
        data_matrix_min_max = (data_matrix_zero_center - np.min(data_matrix_zero_center, axis=0)) / (
                    np.max(data_matrix_zero_center, axis=0) - np.min(data_matrix_zero_center, axis=0))
        return data_matrix_min_max
    else:
        return data_matrix

def select_data_points_from_clusters(clustered_files, K_cluster, geometry_scores, pca_embedding, cluster_folder):
    random.seed(42)
    np.random.seed(42)
    # Create a new scores file that includes the cluster labels and that only includes the selected geometries
    geometry_scores['cluster_label'] = -1
    selected_geometry_scores = []
    selected_geometry_labels = []
    
    for cluster_label in range(len(clustered_files)):
        cluster_files = clustered_files[cluster_label]
        subcluster_folder = os.path.join(cluster_folder, f'Cluster_{cluster_label}')
        cluster_data_matrix = get_data_matrix(subcluster_folder, min_max=True)

        # Perform PCA
        reducer = PCA(n_components=3)
        cluster_embedding = reducer.fit_transform(cluster_data_matrix)

        # Perform KMeans clustering for the selected cluster
        kmeans_cluster = KMeans(n_clusters=K_cluster, random_state=0)
        cluster_class = kmeans_cluster.fit_predict(cluster_embedding)
        clustered_files_cluster = {}
        for l in range(K_cluster):
            clustered_files_cluster[l] = []

        # Assign files to each sub-cluster
        for i in range(len(cluster_class)):
            clustered_files_cluster[cluster_class[i]].append(cluster_files[i])
        selected_geometries_folder = os.path.join(cluster_folder, 'SelectedGeometries')
        os.makedirs(selected_geometries_folder, exist_ok=True)
        
        # Plot KNN with colors for each cluster
        cluster_colors = ['cyan', 'orange', 'purple', 'green', 'pink', 'gray', 'blue', 'magenta', 'yellow', 'black']
        plt.figure()
        plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1], c=[cluster_colors[label] for label in cluster_class])
        
        for sub_cluster_label in range(K_cluster):
            sub_cluster_files = clustered_files_cluster[sub_cluster_label]
            # Select the 3 files with the highest score in the subcluster
            sorted_files = sorted(
                sub_cluster_files,
                key=lambda f: geometry_scores.loc[geometry_scores["pdb_name"] == os.path.basename(f), "total_geometry_score"].values[0],
                reverse=True
            )
            selected_files = sorted_files[:3]
            
            # Copy selected files to output directory
            for selected_file in selected_files:
                new_file_path = os.path.join(selected_geometries_folder, os.path.basename(selected_file))
                # print(f'Copying {selected_file} to {new_file_path}')
                os.system(f'cp {selected_file} {new_file_path}')
                
                # Add data points to the selected_geometry_scores list
                selected_geometry_scores.append(geometry_scores.loc[geometry_scores["pdb_name"] == os.path.basename(selected_file)])
                selected_geometry_labels.append(cluster_label)

            # Modify plot to highlight selected files
            for selected_file in selected_files:
                selected_index = cluster_files.index(selected_file)
                plt.scatter(cluster_embedding[selected_index, 0], cluster_embedding[selected_index, 1], color='red', s=70, alpha=0.3)
        
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'K-nearest neighbors of selected geometries in Cluster {cluster_label}')
        plt.savefig(f'{selected_geometries_folder}/KNN_Cluster_{cluster_label}.png')
        plt.close()
            
    selected_geometry_scores_df = pd.concat(selected_geometry_scores)
    selected_geometry_scores_df['cluster_label'] = selected_geometry_labels
    print('Columns of the selected geometry scores df:')
    print(selected_geometry_scores_df.columns)
    selected_geometry_scores_df.to_csv(os.path.join(cluster_folder, 'SelectedGeometries_Scores.csv'), index=False)
        
               
# *********** Extra functions *****************
def get_pca(data_matrix, n_components=3):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_matrix)
    explained_variance = pca.explained_variance_ratio_
    print('Explained variance: ', explained_variance)
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(principal_components[:, 0], principal_components[:, 1])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of PDB Data')
        plt.grid(True)
        plt.show()
    elif n_components == 3:
        fig = px.scatter_3d(
            principal_components,
            x=0, y=1, z=2,
            title='PCA of PDB Data (3D)',
            labels={'0': 'Principal Component 1', '1': 'Principal Component 2', '2': 'Principal Component 3'}
        )
        # fig.write_html('interactive_pca_plot_min_max.html')
        fig.show()
    elif n_components == 4:
        fig = px.scatter_3d(
            principal_components,
            x=0, y=1, z=2,
            color=principal_components[:, 3],
            title='PCA of PDB Data (3D)',
            labels={'0': 'Principal Component 1', '1': 'Principal Component 2', '2': 'Principal Component 3'}
        )
    return principal_components

def get_kmeans(principal_components, pdb_files, K=7):
    kmeans = KMeans(n_clusters=7, random_state=0)
    cluster_labels = kmeans.fit_predict(principal_components)
    clustered_files = {}
    for cluster_label in range(K):
        clustered_files[cluster_label] = []
    for i in range(len(cluster_labels)):
        clustered_files[cluster_labels[i]].append(pdb_files[i])

    if principal_components.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        for cluster_label in range(K):
            cluster_points = principal_components[cluster_labels == cluster_label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Rescaled Data')
        plt.legend()
        plt.grid(True)
        plt.show()
    elif principal_components.shape[1] == 3:
        fig = px.scatter_3d(
            principal_components,
            x=0, y=1, z=2,
            color=cluster_labels,
            title='PCA of PDB Data (3D)',
            labels={'0': 'Principal Component 1', '1': 'Principal Component 2', '2': 'Principal Component 3'}
        )
        # fig.write_html('interactive_pca_plot_min_max.html')
        fig.show()
    elif principal_components.shape[1] == 4:
        fig = px.scatter_3d(
            principal_components,
            x=0, y=1, z=2,
            color=cluster_labels,
            symbol=principal_components[:, 3],
            title='PCA of PDB Data (3D)',
            labels={'0': 'Principal Component 1', '1': 'Principal Component 2', '2': 'Principal Component 3'}
        )
        # fig.write_html('interactive_pca_plot_min_max.html')
        fig.show()
    return clustered_files

def get_dunn_index(embedding, cluster_labels, K):
    intra_cluster_distances = [np.mean(euclidean_distances(embedding[cluster_labels == k])) for k in range(K)]
    cluster_centers = kmeans.cluster_centers_
    inter_cluster_distances = euclidean_distances(cluster_centers)
    cluster_separation_scores = np.zeros((K, K))
    dunn_index = np.inf
    for i in range(K):
        intra_distance = intra_cluster_distances[i]
        inter_distances = inter_cluster_distances[i]
        for j in range(K):
            score = inter_distances[j] / intra_distance
            cluster_separation_scores[i, j] = score
            if score < dunn_index and score != 0:
                dunn_index = score
                min_clusters = (i, j)
    return dunn_index, min_clusters, cluster_separation_scores

def get_sihuette_scores (embedding, cluster_labels):
    silhouette_vals = silhouette_samples(embedding, cluster_labels)
    cluster_silhouette_scores = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_silhouette_scores:
            cluster_silhouette_scores[label] = []
        cluster_silhouette_scores[label].append(silhouette_vals[i])
    
    average_scores_per_cluster = {label: np.mean(scores) for label, scores in cluster_silhouette_scores.items()}
    # print warning if any of them is lower than 0.5
    for label, score in average_scores_per_cluster.items():
        if score < 0.5:
            print(f'Warning: Average silhouette score for cluster {label} is {score}')
    return average_scores_per_cluster

def get_rmsd(pdb_file1, pdb_file2):
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure("structure1", pdb_file1)
    structure2 = parser.get_structure("structure2", pdb_file2)

    atoms1 = [atom for atom in structure1.get_atoms() if atom.get_name() == 'CA']
    atoms2 = [atom for atom in structure2.get_atoms() if atom.get_name() == 'CA']

    super_imposer = Superimposer()
    super_imposer.set_atoms(atoms1, atoms2)
    super_imposer.apply(structure2.get_atoms())

    return super_imposer.rms

def get_min_rmsd(folder1, folder2, max_iter=4):
    pdb_files1 = [f for f in os.listdir(folder1) if f.endswith('.pdb')]
    pdb_files2 = [f for f in os.listdir(folder2) if f.endswith('.pdb')]
    min_rmsd = 100
    for i in range(4):
        pdb_file1 = random.choice(pdb_files1)
        pdb_file2 = random.choice(pdb_files2)
        pdb_file1_path = os.path.join(folder1, pdb_file1)
        pdb_file2_path = os.path.join(folder2, pdb_file2)
        rmsd = get_rmsd(pdb_file1_path, pdb_file2_path)
        if rmsd < min_rmsd:
            min_rmsd = rmsd
    return min_rmsd

def get_max_rmsd(folder, max_iter=4):
    pdb_files = [f for f in os.listdir(folder) if f.endswith('.pdb')]
    max_rmsd = 0
    for i in range(4):
        pdb_file1 = random.choice(pdb_files)
        pdb_file2 = random.choice(pdb_files)
        while pdb_file1 == pdb_file2:
            pdb_file2 = random.choice(pdb_files)
        pdb_file1_path = os.path.join(folder, pdb_file1)
        pdb_file2_path = os.path.join(folder, pdb_file2)
        rmsd = get_rmsd(pdb_file1_path, pdb_file2_path)
        if rmsd > max_rmsd:
            max_rmsd = rmsd
    return max_rmsd
# *********************************************************

def main(input_geometries_folder, output_cluster_folder, scores_file, K=12, K_cluster=3):
    pdb_files = [f for f in os.listdir(input_geometries_folder) if f.endswith('.pdb')]
    data_matrix = get_data_matrix(input_geometries_folder, min_max=True)
    pdb_files = [os.path.join(input_geometries_folder, f) for f in pdb_files] # Get the full path of the pdb files
    
    reducer = PCA(n_components=5)
    embedding = reducer.fit_transform(data_matrix)
    print('Explained variance: ', reducer.explained_variance_ratio_)

    kmeans = KMeans(n_clusters=K, random_state=0)
    cluster_labels = kmeans.fit_predict(embedding)
    
    # Create folder for each cluster
    os.makedirs(output_cluster_folder, exist_ok=True)
    for cluster_label in range(K):
        os.makedirs(os.path.join(output_cluster_folder, f'Cluster_{cluster_label}'), exist_ok=True)
    
    clustered_files = {}
    for cluster_label in range(K):
        clustered_files[cluster_label] = []
    for i in range(len(cluster_labels)):
        clustered_files[cluster_labels[i]].append(pdb_files[i])
        # Copy the files to the corresponding cluster folder
        new_file_path = os.path.join(output_cluster_folder, f'Cluster_{cluster_labels[i]}/{os.path.basename(pdb_files[i])}')
        os.system(f'cp {pdb_files[i]} {new_file_path}')

    geometry_scores = pd.read_csv(scores_file) # Read the geometry scores for the files
    select_data_points_from_clusters(clustered_files, K_cluster, geometry_scores, embedding, output_cluster_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_geometries_folder",help="Path to the input geometries folder")
    parser.add_argument("--output_cluster_folder", help="Path to the output cluster folder")
    parser.add_argument("--scores_file", help="Path to the CSV file with geometry scores")
    args = parser.parse_args()
    main(args.input_geometries_folder, args.output_cluster_folder, args.scores_file)
