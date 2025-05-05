from pyrosetta import *
from pyrosetta.rosetta.core.select.residue_selector import InterGroupInterfaceByVectorSelector, ChainSelector, OrResidueSelector, NotResidueSelector
import pandas as pd
import argparse
import os
from geometry_functions import *
init()


def run_design_positions_generator (interface, fixed_chains, CB_DISTANCE, EXTENDED_DESIGN, MAX_DISTANCE, PDB_DIR):

    if EXTENDED_DESIGN and MAX_DISTANCE is None:
        raise ValueError("--extended_design requires MAX_DISTANCE to be set (default=12 Å)")

    grp1_str, grp2_str = interface.split('_', 1)
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

            sel1 = OrResidueSelector(*[ChainSelector(c) for c in GROUP1])
            sel2 = OrResidueSelector(*[ChainSelector(c) for c in GROUP2])
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
                ires = [str(i-chain_map[chain]) for i in range(1, pose.size() + 1) if interface_residues[i] and pose.pdb_info().chain(i) == str(chain)]
                interface_residues_dict[chain] = ires

            PDB_NAME = os.path.basename(file).replace(".pdb", "")
            with open(f"{PDB_DIR}/{PDB_NAME}_design_residues.txt", "w") as f:
                f.write(" ".join(interface_residues_dict.keys())) # e.g. f.write("A B C D")
                f.write("\n")

                residue_strings = []
                for chain in interface_residues_dict.keys():
                    residue_strings.append(" ".join(interface_residues_dict[chain])) if interface_residues_dict[chain] else residue_strings.append("")
                f.write(", ".join(residue_strings))

            len_chainA = pose.chain_end(1) - pose.chain_begin(1) + 1
            len_chainB = pose.chain_end(2) - pose.chain_begin(2) + 1
            len_chainC = pose.chain_end(3) - pose.chain_begin(3) + 1

            chAB = OrResidueSelector(ChainSelector('A'), ChainSelector('B'))
            chCD = OrResidueSelector(ChainSelector('C'), ChainSelector('D'))
            interface_selector = InterGroupInterfaceByVectorSelector(chAB, chCD)
            interface_selector.cb_dist_cut(CB_DISTANCE)
            not_interface_selector = NotResidueSelector(interface_selector)
            interface_residues = interface_selector.apply(pose)

            interface_residues_A = [str(i) for i in range(1, pose.size() + 1) if interface_residues[i] and pose.pdb_info().chain(i) == 'A']
            interface_residues_B = [str(i-len_chainA) for i in range(1, pose.size() + 1) if interface_residues[i] and pose.pdb_info().chain(i) == 'B']
            interface_residues_C = [str(i-len_chainA-len_chainB) for i in range(1, pose.size() + 1) if interface_residues[i] and pose.pdb_info().chain(i) == 'C']
            interface_residues_D = [str(i-len_chainA-len_chainB-len_chainC) for i in range(1, pose.size() + 1) if interface_residues[i] and pose.pdb_info().chain(i) == 'D']

            interface_residues_dict = {
                "A": interface_residues_A,
                "B": interface_residues_B,
                "C": interface_residues_C,
                "D": interface_residues_D,
            }
            PDB_NAME = os.path.basename(file).replace(".pdb", "")

            with open(f"{PDB_DIR}/{PDB_NAME}_design_residues.txt", "w") as f:
                f.write("A B C D")
                f.write("\n")

                residue_strings = []
                residue_strings.append(" ".join(interface_residues_dict["A"])) if interface_residues_dict["A"] else residue_strings.append("")
                residue_strings.append(" ".join(interface_residues_dict["B"])) if interface_residues_dict["B"] else residue_strings.append("")
                residue_strings.append(" ".join(interface_residues_dict["C"])) if interface_residues_dict["C"] else residue_strings.append("")
                residue_strings.append(" ".join(interface_residues_dict["D"])) if interface_residues_dict["D"] else residue_strings.append("")
                f.write(", ".join(residue_strings))


    if EXTENDED_DESIGN:
        
        print("************ Make sure your chain numbering starts with 1!!!!!!! ************")

        print ("Designing residues at interface and nearby - using c alpha distance as cutoff")
        files = [f.path for f in os.scandir(PDB_DIR) if f.is_file() and f.path.endswith('.pdb')]
        for file in files:
            print (f'Processing {file}')
            PDB_NAME = os.path.basename(file).replace(".pdb", "")
            _, interface_contacts = analyze_interface (file, min_distance=1, max_distance=MAX_DISTANCE)
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
                    interactions = interactions | interface_contacts[cc1][chain] | interface_contacts[cc2][chain]
            
            print (interface_residues_dict)
            with open(f"{PDB_DIR}/{PDB_NAME}_design_residues.txt", "w") as f:
                f.write(" ".join(interface_residues_dict.keys()))
                f.write("\n")
                residue_strings = [
                    " ".join(interface_residues_dict[c]) if interface_residues_dict[c] else ""
                    for c in interface_residues_dict
                ]
                f.write(", ".join(residue_strings))
            