"""
Scripts used for interaction with pyRosetta packages
"""


import argparse
import os

from pyrosetta import *
from pyrosetta.toolbox import *


def load_pdb(pdb_id, RCSB=False, ATOM=True, CRYS=False):
    """
    :param pdb_id: The four-letter accession code of the desired PDB file
    :type pdb_id: string
    :param RCSB: If True: instead of using local file, downloads file from RCSB
    :param ATOM: Only write ATOM records to disk. Defaults to True.
    :param CRYS: Attempt to extract a monomer from the target PDB. Defaults to False.
    """
    pdb_id = pdb_id.upper()
    if RCSB:
        pose = pose_from_rcsb(pdb_id, ATOM, CRYS)
        pose_clean = pose_from_pdb(os.path.join(dir_path, pdb_id+'.clean.pdb'))
    else:
        pose = pose_from_pdb(os.path.join(dir_path, pdb_id+'.pdb'))
        cleanATOM(os.path.join(dir_path, pdb_id+'.pdb'))
        pose_clean = pose_from_pdb(os.path.join(dir_path, pdb_id+'.clean.pdb'))

    return pose, pose_clean

dir_path = os.getcwd()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')  # TODO: Write description
    parser.add_argument('filename', type=str, help='name of PDB file')
    args = parser.parse_args()

    filename = args.filename.upper()

    init()
    pose, pose_clean = load_pdb(filename, RCSB=False)
    print(pose.sequence())
    print(pose_clean.sequence())

