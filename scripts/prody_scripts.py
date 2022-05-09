# Requirements:
# Latest version of ProDy, Matplotlib, VMD and NAMD

from prody import *
from pylab import *
import argparse
from os.path import exists


def sample_load_pdb(pdb_id):  # TODO: Change name
    pdb_id = pdb_id.upper()
    # TODO: Check if file exists

    if pdb_id not in proteins:
        proteins[pdb_id] = parsePDB(pdb_id)
    # TODO: Save PDB somewhere else


def show_protein(pdb_id):  # TODO: It works only in interactive viewer?
    pdb_id = pdb_id.upper()
    prody.showProtein(proteins[pdb_id])
    legend()


def calc_anm(protein_name, normal_modes=20):  # in tutorial they used 3 n_modes #TODO: ASK
    protein_name = protein_name.upper()
    protein_ca = proteins[protein_name].ca

    # instantiate an ANM object
    protein_anm = ANM(protein_ca)
    # proteins[protein_name+'_anm'] = protein_anm

    # build Hessian matrix
    protein_anm.buildHessian(protein_ca)

    protein_anm.calcModes(n_modes=normal_modes)

    proteins[protein_name+'_anm'] = protein_anm

#############################################################################################


parser = argparse.ArgumentParser(description='')  # TODO: Write description
parser.add_argument('filename', type=str, help='name of PDB file')

args = parser.parse_args()

filename = args.filename
proteins = {}

# TESTING SITE ##########################################################################
sample_load_pdb(filename)
print(proteins)
# print(proteins.keys())
# show_protein(filename)

calc_anm(filename)
print(proteins)

# showSqFlucts(proteins[filename.upper()+'_anm']) #TODO: Doesn't work outside of iPython

writeNMD(filename.upper()+'_anm.nmd', proteins[filename.upper()+'_anm'], proteins[filename.upper()].ca)
viewNMDinVMD('1P38_anm.nmd')
