from pyrosetta import *
from pyrosetta.rosetta.core.select.residue_selector import InterGroupInterfaceByVectorSelector, ChainSelector, OrResidueSelector, NotResidueSelector
import pandas as pd
import argparse
import os
from geometry_functions import *
init()

parser = argparse.ArgumentParser(
    description="Extract designable interface residues for any two groups of chains."
)
parser.add_argument('--interface', type=str, required=True, help=("Interface spec as GROUP1_GROUP2, e.g. A_BC or AB_CD or A_D "))
parser.add_argument('--fixed_chains', type=str, default='', help="Chain IDs to keep fixed (unmutated), e.g. A,B")
parser.add_argument('--cb_distance', type=float, default=11.0, help='Distance between C-beta atoms of two residues to be considered as interacting')
parser.add_argument('--extended_design', action='store_true', help='If flag is present, design residues will be extended to 15A contacts')
parser.add_argument('--max_distance', type=float, default=12, help='Max distance for extended design')
parser.add_argument('--pdb_dir', type=str, default='../Outputs-GeometryAnalizer/', help='Directory where PDB files are located')
args = parser.parse_args()

if args.extended_design and args.max_distance is None:
    parser.error("--extended_design recommends --max_distance to be set. Default is 12 A")

# PARAMETERS
PDB_DIR = args.pdb_dir
CB_DISTANCE = args.cb_distance
EXTENDED_DESIGN = args.extended_design
MAX_DISTANCE = args.max_distance

grp1_str, grp2_str = args.interface.split('_', 1)
GROUP1 = list(grp1_str)
GROUP2 = list(grp2_str)
FIXED_CHAINS = [x.strip() for x in args.fixed_chains.split(',')]

cutoff = MAX_DISTANCE if EXTENDED_DESIGN else CB_DISTANCE
mode   = "extended interface (C alpha distance)" if EXTENDED_DESIGN else "interface (C beta distance)"
print(f"Mode = {mode}, cutoff = {cutoff:.1f} Å")
print(f"Designing interface between {GROUP1} and {GROUP2}, keeping {FIXED_CHAINS} fixed")

if not EXTENDED_DESIGN:
    print ("Designing residues at interface only")
    files = [f.path for f in os.scandir(PDB_DIR) if f.is_file() and f.path.endswith('.pdb')]
    for file in files:
        print (f'Processing {file}')
        pose = pose_from_pdb(file)
        start_pose = Pose()
        start_pose.assign(pose)

        sel1 = group_selector(GROUP1)
        sel2 = group_selector(GROUP2)

        interface_selector = InterGroupInterfaceByVectorSelector(sel1, sel2)
        interface_selector.cb_dist_cut(CB_DISTANCE)
        not_interface_selector = NotResidueSelector(interface_selector)
        interface_residues = interface_selector.apply(pose) # Boolean vector

        # Map the chain to the index of where it begins
        chain_map = {}
        for i in range(1, pose.num_chains()+1):
            ch = pose.pdb_info().chain(pose.chain_begin(i))
            chain_map[ch] = pose.chain_begin(i)
        print(f"Chain begin index: {chain_map}")

        # interface_residues_dict of designable residues
        interface_residues_dict = {
            **{c: [] for c in GROUP1}, # ** operator unpacks a dict into key:value
            **{c: [] for c in GROUP2}
            } # All unpacked key:values are merged

        for chain in interface_residues_dict.keys():
            if chain in FIXED_CHAINS:
                continue
            ires = [str(i-chain_map[chain] + 1) for i in range(1, pose.size() + 1) if interface_residues[i] and pose.pdb_info().chain(i) == str(chain)]
            interface_residues_dict[chain] = ires

        PDB_NAME = os.path.basename(file).replace(".pdb", "")
        with open(f"{PDB_DIR}/{PDB_NAME}_design_residues.txt", "w") as f:
            f.write(" ".join(interface_residues_dict.keys())) # e.g. f.write("A B C D")
            f.write("\n")

            residue_strings = []
            for chain in interface_residues_dict.keys():
                residue_strings.append(" ".join(interface_residues_dict[chain])) if interface_residues_dict[chain] else residue_strings.append("")
            f.write(", ".join(residue_strings))


if EXTENDED_DESIGN:
    
    print("************ Make sure your chain numbering starts with 1!!!!!!! ************")

    print ("Designing residues at interface and nearby - using c alpha distance as cutoff")
    files = [f.path for f in os.scandir(PDB_DIR) if f.is_file() and f.path.endswith('.pdb')]
    for file in files:
        print (f'Processing {file}')
        PDB_NAME = os.path.basename(file).replace(".pdb", "")
        pairs_to_omit = (
            list(permutations(GROUP1, 2)) +
            list(permutations(GROUP2, 2))
        )
        _, interface_contacts = analyze_interface (file, min_distance=1, max_distance=MAX_DISTANCE, omit_chain_pairs=pairs_to_omit)
        interface_residues_dict = { c: [] for c in (GROUP1 + GROUP2) }

        for chain in interface_residues_dict:
            if chain in FIXED_CHAINS:
                continue

            interactions = set()
            for other_chain in (GROUP1 + GROUP2):
                if other_chain == chain:
                    continue
                cc1 = chain + other_chain
                cc2 = other_chain + chain
                interactions |= interface_contacts.get(cc1, {}).get(chain, set())
                interactions |= interface_contacts.get(cc2, {}).get(chain, set())
            interface_residues_dict[chain] = [i+1 for i in interactions]

        print (interface_residues_dict)
        with open(f"{PDB_DIR}/{PDB_NAME}_design_residues.txt", "w") as f:
            f.write(" ".join(interface_residues_dict.keys()))
            f.write("\n")

            residue_strings = []
            for c in interface_residues_dict:
                residue_strings.append(" ".join(str(i) for i in interface_residues_dict[c]))

            f.write(", ".join(residue_strings))