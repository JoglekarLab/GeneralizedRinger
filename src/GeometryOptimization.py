import argparse
import os
import numpy as np
from scipy.optimize import basinhopping, dual_annealing, shgo, brute
from geometry_functions import *
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print('Timestamp: ', timestamp)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Optimize monomer arrangements in a protein ring.')
parser.add_argument('--nSym', type=int, required=True, help='Number of monomers in the ring')
parser.add_argument('--radius', type=int, required=True, help='Radius of the ring')
parser.add_argument('--monomer_pdb_path', type=str, required=True, help='Path to the monomer PDB file, e.g., /path/to/monomer.pdb')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files, e.g., /path/to/output')
parser.add_argument('--method', type=str, required=True, choices=['annealing', 'shgo', 'brute', 'basinhopping'], help='Optimization method to use')
parser.add_argument('--fixed_rotation', action='store_true', help='Whether to fix the rotation of the monomers')
parser.add_argument('--rotx', type=int, default=0, help='Rotation around the x axis, required if fixed_rotation is True')
parser.add_argument('--roty', type=int, default=0, help='Rotation around the y axis, required if fixed_rotation is True')
parser.add_argument('--rotz', type=int, default=0, help='Rotation around the z axis, required if fixed_rotation is True')
args = parser.parse_args()

# Extract the monomer name from the pdb path
MONOMER_NAME = os.path.basename(args.monomer_pdb_path).replace('.pdb', '')
nSym = args.nSym
radius = args.radius
FIXED_ROTATION = args.fixed_rotation
nSym_range = [9, 21]  # Only applicable if nSym is not specified
radius_range = [100, 240]  # Only applicable if radius is not specified
filename = args.monomer_pdb_path

# Ensure output directory exists
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)


if np.isnan(args.nSym) and np.isnan(args.radius):
    subfolder = f'totalScan'
elif np.isnan(args.nSym) and FIXED_ROTATION:
    subfolder = f'r{radius}_fixedrot{rotx}_{roty}_{rotz}'
elif np.isnan(args.radius) and FIXED_ROTATION:
    subfolder = f'{nSym}_fixedrot{rotx}_{roty}_{rotz}'
elif np.isnan(args.nSym):
    subfolder = f'{radius}_fixedRadiusScan'
elif np.isnan(args.radius):
    subfolder = f'{nSym}_FixednSymScan'
else:
    subfolder = f'Fixed_nSym{nSym}_radius{radius}'
subfolder_path = os.path.join(output_dir, subfolder)
if not os.path.exists(subfolder_path):
    os.makedirs(subfolder_path, exist_ok=True)

# Get coordinates by chain
coords_by_chain = CA_coords(args.monomer_pdb_path)

# Radius correction
all_coords = np.concatenate(list(coords_by_chain.values()))
x_len, y_len, z_len = all_coords.max(axis=0) - all_coords.min(axis=0)
radius_correction = (x_len + y_len) / 4
corrected_radius = args.radius + radius_correction
print('Structure calculated with radius: ', corrected_radius)

def objective_function(x):
    '''
    Args:
        x (list): list of parameters to optimize given as [nSym, radius, rotx, roty, rotz]
    '''
    global subfolder_path2
    if np.isnan(args.nSym) and np.isnan(args.radius):
        nSym, radius = round(x[0]), x[1]
        if not FIXED_ROTATION:
            rotx, roty, rotz = x[2], x[3], x[4]
    elif np.isnan(args.nSym) and FIXED_ROTATION:
        radius, rotx, roty, rotz = args.radius, args.rotx, args.roty, args.rotz
        nSym = round(x[0])
    elif np.isnan(args.radius) and FIXED_ROTATION:
        nSym, rotx, roty, rotz = args.nSym, args.rotx, args.roty, args.rotz
        radius = x[0]
    elif np.isnan(args.nSym):
        radius = args.radius
        nSym, rotx, roty, rotz = round(x[0]), x[1], x[2], x[3]
    elif np.isnan(args.radius):
        nSym = args.nSym
        radius, rotx, roty, rotz = x[0], x[1], x[2], x[3]
    else:
        nSym, radius = args.nSym, args.radius
        rotx, roty, rotz = x[0], x[1], x[2]
    score, n_contacts, _, _ = get_score_and_contacts(filename, corrected_radius, rotx, roty, rotz, nSym)
    n_score = score * n_contacts  
    if np.isfinite(score):
        with open(os.path.join(subfolder_path2, f'Optimization_{timestamp}.csv'), 'a') as f:
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(nSym, radius, corrected_radius, rotx, roty, rotz, n_score, score, n_contacts))  
    elif np.isnan(score):
        score = -np.inf
    return score * n_contacts

def optimize():
    global subfolder_path2
    # VARIABLES TO BE OPTIMIZED
    if np.isnan(args.nSym) and np.isnan(args.radius):
        if FIXED_ROTATION:
            print(f'Performing a scan at fixed rotations {rotx}, {roty}, {rotz}')
            bounds =  [(nSym_range[0], nSym_range[-1]), (radius_range[0], radius_range[-1])]
            x0 = np.array([np.random.uniform(bounds[0][0], bounds[0][1]), np.random.uniform(bounds[1][0], bounds[1][1])])
            print(f'Bounds:\n nSym: {bounds[0]}\n radius: {bounds[1]}')
        else:
            print('Performing a total scan')
            bounds = [(nSym_range[0], nSym_range[-1]), (radius_range[0], radius_range[-1]), (0, 360), (0, 360), (0, 360)]
            x0 = np.array([np.random.uniform(bounds[0][0], bounds[0][1]), np.random.uniform(bounds[1][0], bounds[1][1]), np.random.uniform(bounds[2][0], bounds[2][1]), np.random.uniform(bounds[3][0], bounds[3][1]), np.random.uniform(bounds[4][0], bounds[4][1])])
            print(f'Bounds:\n nSym: {bounds[0]}\n radius: {bounds[1]}\n rotx: {bounds[2]}\n roty: {bounds[3]}\n rotz: {bounds[4]}')
    elif np.isnan(args.nSym) and FIXED_ROTATION:
        print(f'Performing a fixed rotation scan, with rotations: {rotx}, {roty}, {rotz} and radius {args.radius}')
        bounds = [(nSym_range[0], nSym_range[-1])]
        x0 = np.array([np.random.uniform(bounds[0][0], bounds[0][1])])
        print(f'Bounds:\n nSym: {bounds[0]}')
    elif np.isnan(args.radius) and FIXED_ROTATION:
        print(f'Performing a fixed rotation scan, with rotations: {rotx}, {roty}, {rotz} and nSym {args.nSym}')
        bounds = [(radius_range[0], radius_range[-1])]
        x0 = np.array([np.random.uniform(bounds[0][0], bounds[0][1])])
        print(f'Bounds:\n radius: {bounds[0]}')
    elif np.isnan(args.nSym):
        print('Performing a fixed radius scan, with radius: ', args.radius)
        bounds = [(nSym_range[0], nSym_range[-1]), (0, 360), (0, 360), (0, 360)]
        x0 = np.array([np.random.uniform(bounds[0][0], bounds[0][1]), np.random.uniform(bounds[1][0], bounds[1][1]), np.random.uniform(bounds[2][0], bounds[2][1]), np.random.uniform(bounds[3][0], bounds[3][1])])
        print(f'Bounds:\n nSym: {bounds[0]}\n rotx: {bounds[1]}\n roty: {bounds[2]}\n rotz: {bounds[3]}')
    elif np.isnan(args.radius):
        print('Performing a fixed nSym scan, with nSym: ', args.nSym)
        bounds = [(radius_range[0], radius_range[-1]), (0, 360), (0, 360), (0, 360)]
        x0 = np.array([np.random.uniform(bounds[0][0], bounds[0][1]), np.random.uniform(bounds[1][0], bounds[1][1]), np.random.uniform(bounds[2][0], bounds[2][1]), np.random.uniform(bounds[3][0], bounds[3][1])])
        print(f'Bounds:\n radius: {bounds[0]}\n rotx: {bounds[1]}\n roty: {bounds[2]}\n rotz: {bounds[3]}')
    else:
        print('Performing a fixed nSym and radius scan, with nSym: ', args.nSym, ' and radius: ', args.radius)
        bounds = [(0, 360), (0, 360), (0, 360)]
        x0 = np.array([np.random.uniform(bounds[0][0], bounds[0][1]), np.random.uniform(bounds[1][0], bounds[1][1]), np.random.uniform(bounds[2][0], bounds[2][1])])
        print(f'Bounds:\n rotx: {bounds[0]}\n roty: {bounds[1]}\n rotz: {bounds[2]}')

    # OPTIMIZATION
    method = args.method
    
    if method == "annealing":
        print('Optimizing using dual annealing...')
        subfolder = 'DualAnnealing'
        subfolder_path2 = os.path.join(subfolder_path, subfolder)
        if not os.path.exists(subfolder_path2):
            os.makedirs(subfolder_path2)
        with open(os.path.join(subfolder_path2, f'Optimization_{timestamp}.csv'), 'a') as f:
            f.write('nSym,radius,corrected_radius,rotx,roty,rotz,score*n_contacts,score,n_contacts\n')
        result = dual_annealing(objective_function, bounds, maxiter=10000, initial_temp=5230)
        print('Optimization with dual annealing finished')
        print('Optimal parameters: ', result.x)
        print('Optimal score: ', -result.fun)
    elif method == "shgo":
        print('Optimizing using SHGO...')
        subfolder = 'SHGO'
        subfolder_path2 = os.path.join(subfolder_path, subfolder)
        if not os.path.exists(subfolder_path2):
            os.makedirs(subfolder_path2)
        with open(os.path.join(subfolder_path2, f'Optimization_{timestamp}.csv'), 'a') as f:
            f.write('nSym,radius,corrected_radius,rotx,roty,rotz,score*n_contacts,score,n_contacts\n')
        result = shgo(objective_function, bounds, n=100, iters=15, sampling_method='sobol')
        print('Optimization with SHGO finished')
        print('Optimal parameters: ', result.x)
        print('Optimal score: ', -result.fun)
    elif method == "brute":
        print('Optimizing using brute force...')
        subfolder = 'BruteForce'
        subfolder_path2 = os.path.join(subfolder_path, subfolder)
        if not os.path.exists(subfolder_path2):
            os.makedirs(subfolder_path2)
        with open(os.path.join(subfolder_path2, f'Optimization_{timestamp}.csv'), 'a') as f:
            f.write('nSym,radius,corrected_radius,rotx,roty,rotz,score*n_contacts,score,n_contacts\n')
        result = brute(objective_function, bounds, Ns=72, full_output=True, finish=None)
        print('Optimization with brute force finished')
    elif method == "basinhopping":
        print('Optimizing using basin hopping...')
        subfolder = 'basinhopping'
        subfolder_path2 = os.path.join(subfolder_path, subfolder)
        if not os.path.exists(subfolder_path2):
            os.makedirs(subfolder_path2)
        with open(os.path.join(subfolder_path2, f'Optimization_{timestamp}.csv'), 'a') as f:
            f.write('nSym,radius,corrected_radius,rotx,roty,rotz,score*n_contacts,score,n_contacts\n')
        minimizer_kwargs = {"bounds": bounds}
        result = basinhopping(objective_function, x0, minimizer_kwargs=minimizer_kwargs, niter=5000)
        print('Optimization with basin hopping finished')
        print('Optimal parameters: ', result.x)
        print('Optimal score: ', -result.fun)
    else:
        raise ValueError(f"Unknown optimization method: {method}")

optimize()