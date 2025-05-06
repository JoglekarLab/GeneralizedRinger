from itertools import product, combinations, permutations
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO
from matplotlib import pyplot as plt
import pandas as pd
import os
from scipy.stats import norm, kstest
from Bio.PDB import PDBParser, Structure, Model, Chain, Residue, Atom, PDBIO
# import nglview as nv


# import warnings
# warnings.filterwarnings('ignore')
# # warnings.resetwarnings()

def CA_coords(filename):
    '''
    Get the coordinates of the alpha carbon atoms of a protein from a PDB file.
    Args:
        filename (str): path to the pdb file
    Returns:
        coords_by_chain (dict): dictionary with the coordinates of the alpha carbon atoms for each chain
    '''
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", filename)
    coords_by_chain = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id not in coords_by_chain:
                coords_by_chain[chain_id] = np.array([atom.get_coord() for atom in chain.get_atoms() if atom.name == 'CA'])
    return coords_by_chain

def transform_points(points, radius, rotx, roty, rotz, nSym, whole_ring = False):
    '''
    Creates a ring from a set of points.
    '''
    rot1 = R.from_euler('x', 90, degrees=True)
    rot2 = R.from_euler('xyz', [rotx, roty, rotz], degrees=True)
    points = rot1.apply(points)
    points = rot2.apply(points)

    result_points = []

    x = 2 # We only need to create two protomers of the ring to analyze the interface
    if whole_ring:
         x = nSym
    for index in range(x):
            angle = 360 / nSym * index
            rot3 = R.from_euler('z', angle, degrees=True)
            transformed_points = rot3.apply(points)
            translation = np.array([
                radius * np.cos(np.deg2rad(angle)),
                radius * np.sin(np.deg2rad(angle)),
                0
            ])
            transformed_points += translation
            result_points.append(transformed_points)
    return result_points

def get_rotational_space(xRot_range, yRot_range, zRot_range, increments = 5):
    rot_space = np.array(tuple(product(
        range(*xRot_range, increments),
        range(*yRot_range, increments),
        range(*zRot_range, increments)
    )))
    return rot_space

def get_score (x, threshold = 6):
    '''
    Scores the interface based on the distances between the chains.
    Args:
        x(list): list of distances
        threshold(float): threshold distance to apply a penalty to the score
    '''
    y = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < threshold:
            y[i] = -0.5*(x[i]-7.55)**2  - 4*(abs(x[i]-threshold))
            y[i] = np.exp(y[i])
        else:
            y[i] = -0.5*(x[i]-7.55)**2
            y[i] = np.exp(y[i])
    return np.mean(y)

def get_contacts(transformed_coords_by_chain, min_distance=5, max_distance=10):
    '''
    Computes the distances between A0 and A1, A0 and B1, B0 and A1, and B0 and B1,
    then returns the minimum distance between each pair of points.
    '''
    distances = []
    contacts_index = {'A0': [], 'A1': [], 'B0': [], 'B1': []}

    pairs = [
        ('A0', 'A1', transformed_coords_by_chain['A'][0], transformed_coords_by_chain['A'][1]),
        ('A0', 'B1', transformed_coords_by_chain['A'][0], transformed_coords_by_chain['B'][1]),
        ('B0', 'A1', transformed_coords_by_chain['B'][0], transformed_coords_by_chain['A'][1]),
        ('B0', 'B1', transformed_coords_by_chain['B'][0], transformed_coords_by_chain['B'][1]),
    ]

    for label_a, label_b, points_a, points_b in pairs:
        for index_a, point_a in enumerate(points_a):
            distance_row = cdist(point_a.reshape(1, -1), points_b)[0]
            min_distance_val = np.min(distance_row)
            min_index = np.argmin(distance_row)

            if min_distance_val < min_distance:
                return np.nan, np.nan
            if min_distance_val < max_distance:
                distances.append(min_distance_val)
                contacts_index[label_a].append(index_a)
                contacts_index[label_b].append(min_index)

    # Remove duplicates from each key in contacts_index
    for key in contacts_index:
        contacts_index[key] = list(set(contacts_index[key]))
    return distances, contacts_index

def get_score_and_contacts(pdb_path, radius, rotx, roty, rotz, nSym):
    coords_by_chain = CA_coords(pdb_path)
    transformed_coords_by_chain = {}
    for chain_id, points in coords_by_chain.items():
        transformed_points = transform_points(points, radius, rotx, roty, rotz, nSym)
        transformed_coords_by_chain[chain_id] = transformed_points

    distances, contacts_index = get_contacts(transformed_coords_by_chain)
    if distances is not np.nan:
        score = get_score(distances)
        n_contacts = sum([len(contacts_index[key]) for key in contacts_index])
    else:
        score = np.nan
        n_contacts = np.nan

    return score, n_contacts, contacts_index, distances

def get_total_score (pdb_path, radius, rotx, roty, rotz, nSym):
    score, n_contacts, _, _ = get_score_and_contacts(pdb_path, radius, rotx, roty, rotz, nSym)
    total_score = score*n_contacts
    total_score = np.nan_to_num(total_score)
    return total_score

def interchain_contacts(pdb_path, min_distance=5, max_distance=10, ignore_close_contacts=False):
    '''
    Get the contacts between chains in any given pdb file and score the interface
    '''
    coords_by_chain = CA_coords(pdb_path)
    print(f'Chains in pdb file: {coords_by_chain.keys()}')
    distances = []
    contacts_index = {key: [] for key in coords_by_chain.keys()} # create dictionary with empty lists for each chain

    for chain_A, chain_B in product(coords_by_chain.keys(), repeat=2):
        if chain_A == chain_B:
            continue
        for index_A, point_A in enumerate(coords_by_chain[chain_A]):
            distances_A = cdist(point_A.reshape(1, -1), coords_by_chain[chain_B])
            min_distance_A = np.min(distances_A)
            min_index_A = np.argmin(distances_A)
            if not ignore_close_contacts:
                if min_distance_A < min_distance:
                    print(f'Too close contact between {chain_A} and {chain_B} with a distance of {min_distance_A}')
                    continue
            if min_distance_A < max_distance:
                distances.append(min_distance_A)
                contacts_index[f'{chain_A}'].append(index_A)
                contacts_index[f'{chain_A}'].append(min_index_A)

    # Remove duplicates from each key in contacts_index
    for key in contacts_index:
        contacts_index[key] = list(set(contacts_index[key]))

    score = get_score(distances)
    n_contacts = sum([len(contacts_index[key]) for key in contacts_index])

    return distances, contacts_index, score, n_contacts


################
# General functions for interface analysis

def get_all_pairs (pdb_path, omit_chain_pairs = []):
    '''
    Gets all the possible pairs of chains from a pdb file except the ones specified in omit_chains_pairs.
    Args:
        pdb_path (str): path to the pdb file
    Returns:
        pairs (list): list of tuples with the pairs of chains to compute the distances
            e.g. [('A', 'B', array([[  0. ,   0. ,   0. ], [  0. ,   0. ,   1. ], ...]), array([[ 2 ,   2.4 ,   0. ], [  0. ,   2 ,   1. ], ...]))]
    '''   
    CA_coords_by_chain = CA_coords(pdb_path) # Gets a dictionary with the coordinates of the alpha carbon atoms for each chain
    chains = list(CA_coords_by_chain.keys())
    pairs = list(permutations(chains, 2)) # Gets all the possible pairs of chains
    pairs = [pair for pair in pairs if pair not in omit_chain_pairs and tuple(reversed(pair)) not in omit_chain_pairs]

    print ("Pairs of chains to compute distances:")
    for i in range(len(pairs)):
        pairs[i] = (pairs[i][0], pairs[i][1], CA_coords_by_chain[pairs[i][0]], CA_coords_by_chain[pairs[i][1]])
        print (pairs[i][0], pairs[i][1], 'lengths:', len(CA_coords_by_chain[pairs[i][0]]), len(CA_coords_by_chain[pairs[i][1]])) # Make sure the pairs considered are correct
    return pairs

def general_get_contacts(pairs, min_distance=4, max_distance=10):
    '''
    Computes the distances between the given pairs of chains and returns the minimum distance between each pair of points.
    Args:
        pairs (list): list of tuples with the pairs of chains to compute the distances
            e.g. [('A', 'B', array([[  0. ,   0. ,   0. ], [  0. ,   0. ,   1. ], ...]), array([[ 2 ,   2.4 ,   0. ], [  0. ,   2 ,   1. ], ...]))]
        min_distance (float): minimum distance between two points to be considered a contact
        max_distance (float): maximum distance between two points to be considered a contact
    Returns:
        distances (list): list of minimum distances between each pair of points (contacts)
        contacts_index (dict): dictionary with the indices of the interface residues for each chain
        n_contacts (int): number of contacts
    '''
    distances = []
    contacts_index = {}
    
    for label_a, label_b, points_a, points_b in pairs:
        if label_a not in contacts_index:
            contacts_index[label_a] = []
        if label_b not in contacts_index:
            contacts_index[label_b] = []

        # compute the distances between the points of the two chains
        for index_a, point_a in enumerate(points_a):
            distance_row = cdist(point_a.reshape(1, -1), points_b)[0]
            min_distance_val = np.min(distance_row)
            min_index = np.argmin(distance_row)

            if min_distance_val < min_distance:
                print(f"CLASH WARNING: Minimum distance between chains {label_a} and {label_b} is {min_distance_val}, being less than the minimum distance threshold.")
                return np.nan, np.nan, np.nan
            if min_distance_val < max_distance:
                distances.append(min_distance_val)
                contacts_index[label_a].append(index_a)
                contacts_index[label_b].append(min_index)

    # make contacts_index a set so that the indices are unique
    for label, indices in contacts_index.items():
        contacts_index[label] = set(indices)

    n_contacts = 0
    for chain, index in contacts_index.items():
        n_contacts += len(index)

    return distances, contacts_index, n_contacts

def analyze_interface (pdb_path, min_distance, max_distance, omit_chain_pairs=[]):
    '''
    Returns a dataframe with the interface scores for each pair of chains in the pdb file.
    Args:
        pdb_path (str): path to the pdb file
        omit_chain_pairs (list): list of tuples with the pairs of chains to omit. e.g. [('A', 'B'), ('C', 'D')]
    Returns:
        scores_df (dataframe): dataframe with the scores for each pair of chains
    '''
    print(f'Analyzing PDB file {os.path.basename(pdb_path)}\n')
    scores_df = pd.DataFrame(columns=['pdb_name', 'chain1', 'chain2', 'per_contact_score', 'n_contacts', 'total_score'])
    pairs = get_all_pairs(pdb_path, omit_chain_pairs)
    contacts = {}
    for pair in pairs:
        distances, contacts_index, n_contacts = general_get_contacts([pair], min_distance, max_distance)
        score = get_score(distances)
        total_score = score * n_contacts
        scores_df.loc[len(scores_df)] = [os.path.basename(pdb_path), pair[0], pair[1], score, n_contacts, total_score]
        contacts[pair[0] + pair[1]] = contacts_index
    return scores_df, contacts

#########################################
# Create ring

def create_pdb_from_transformed_coords(pdb_path, transformed_coords_by_chain, output_path):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_path)
    new_structure = Structure.Structure("transformed")

    for model in structure:
        new_model = Model.Model(model.id)
        new_structure.add(new_model)
        for chain in model:
            chain_id = chain.id
            if chain_id in transformed_coords_by_chain:
                new_chain = Chain.Chain(chain_id)
                new_model.add(new_chain)
                residue_index = 0
                for residue in chain:
                    new_residue = Residue.Residue(residue.id, residue.resname, residue.segid)
                    new_chain.add(new_residue)
                    for atom in residue:
                        if atom.name == 'CA':
                            # Use transformed coordinates
                            new_coords = transformed_coords_by_chain[chain_id][residue_index]
                            new_atom = Atom.Atom(atom.name, new_coords, atom.bfactor, atom.occupancy, atom.altloc, 
                                                 atom.fullname, atom.serial_number, atom.element)
                            new_residue.add(new_atom)
                    residue_index += 1

    # Write new structure to file
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_path)

def create_ring (pdb_path, radius, rotx, roty, rotz, nSym, whole_ring=True):
    view = nv.NGLWidget()

    coords_by_chain = CA_coords(pdb_path)
    all_coords = np.concatenate(list(coords_by_chain.values()))
    x_len, y_len, z_len = all_coords.max(axis=0)-all_coords.min(axis=0)
    radius_correction = (x_len+y_len)/4
    corrected_radius = radius + radius_correction
    print("radius correction = {}".format(radius_correction))

    # transform the coordinates
    transformed_coords_by_chain = {}
    for chain_id, points in coords_by_chain.items():
        transformed_points = transform_points(points, corrected_radius, rotx, roty, rotz, nSym, whole_ring=whole_ring)
        transformed_coords_by_chain[chain_id] = transformed_points

    # Create a new PDB file with the transformed coordinates
    pdb_name = os.path.basename(pdb_path).split('.')[0]
    for i in range(nSym):
        tr_coords = {}
        tr_coords['A'] = transformed_coords_by_chain['A'][i]
        tr_coords['B'] = transformed_coords_by_chain['B'][i]
        output_path = "../Outputs-GeometryAnalizer/transformed_{}_{}.pdb".format(pdb_name, i+1)
        create_pdb_from_transformed_coords(pdb_path, tr_coords, output_path)
        
        transformed_structure = nv.FileStructure(output_path)
        view.add_component(transformed_structure)

    # remove the transformed pdb files
    for i in range(nSym):
        output_path = "../Outputs-GeometryAnalizer/transformed_{}_{}.pdb".format(pdb_name, i+1)
        os.remove(output_path)

    return view
