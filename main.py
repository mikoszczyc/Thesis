from pyrosetta import *
from pyrosetta.toolbox import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    init()  # initialize pyRosetta

    #  Load PDB
    ###  BEGIN SOLUTION

    # pose = pose_from_pdb("inputs/5tj3.pdb")  # fetch from local PDB file
    # cleanATOM('inputs/5tj3.pdb')
    # pose_clean = pose_from_pdb('inputs/5tj3.clean.pdb')
    #
    # # ALTERNATIVELY:
    # pose = pose_from_rcsb("5TJ3")  # download PDB from RCSB and automatically clean PDB

    ###  END SOLUTION