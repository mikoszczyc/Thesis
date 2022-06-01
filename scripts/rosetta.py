"""
Scripts used for interaction with pyRosetta packages
"""


from pyrosetta import *
from pyrosetta.toolbox import *

def rosetta_load_pdb(pdb_code, RCSB=False, ATOM=True, CRYS=False):
    """


    :param pdb_code: The four-letter accession code of the desired PDB file
    :type pdb_code: string
    :param RCSB: If True: instead of using local file, downloads file from RCSB
    :param ATOM: Only write ATOM records to disk. Defaults to True.
    :param CRYS: Attempt to extract a monomer from the target PDB. Defaults to False.
    """
    pdb_code = pdb_code.upper()
    if RCSB:
        pose_from_rcsb(pdb_code, ATOM, CRYS)
    else:
        pose_from_pdb(pdb_code)