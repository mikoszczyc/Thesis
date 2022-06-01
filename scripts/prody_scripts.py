# Requirements:
# Latest version of ProDy, Matplotlib, VMD and NAMD

from prody import *
from pylab import *

import os
import matplotlib
import matplotlib.pyplot as plt
import argparse


def sample_load_pdb(pdb_id):  # TODO: Change name
    pdb_id = pdb_id.upper()

    if pdb_id not in proteins:
        protein = parsePDB(pdb_id)
        proteins[pdb_id] = protein.protein

    # TODO: load all types of files that exist (ag, _anm, _ca.anm, _ext.nma) (anything else?)
    if os.path.exists(pdb_id + '.ag.npz'):
        proteins[pdb_id] = loadAtoms(pdb_id+'.ag.npz')

    if os.path.exists(pdb_id+'_ca.anm.npz'):
        proteins[pdb_id+'_anm'] = loadModel(pdb_id+'_ca.anm.npz')

    if os.path.exists(pdb_id+'_ext.nma.npz'):
        proteins[pdb_id+'_anm_ext'] = loadModel(pdb_id+'_ext.nma.npz')


def show_protein(pdb_id):
    pdb_id = pdb_id.upper()
    prody.showProtein(proteins[pdb_id])
    legend()
    plt.show()


def anm_calc_modes(pdb_id, n_modes=20, zeros=False, turbo=True):  # in tutorial they used 3 n_modes (default: 20)
    """
    Perform ANM (Anisotropic Network Model) calculations and retrieve normal mode data.
    Creates ANM instance that stores Hessian matrix and normal mode data.
    Normal mode data describes intrinsic dynamics of the protein structure.

    :param pdb_id: The four-letter accession code of the desired PDB file
    :type pdb_id: string
    :param n_modes: Number of non-zero eigenvalues/vectors to calculate. Choose None or 'all' to calculate all modes.
    :type n_modes: int or None, default is 20
    :param zeros: If `True`, modes with zero eigenvalues will be kept.
    :type zeros: bool
    :param turbo: Use faster but more memory intensive calculation mode.
    :type turbo: bool
    :return: ANM object
    """

    pdb_id = pdb_id.upper()
    # TODO: Check if needed objects exist in proteins[]

    protein_ca = proteins[pdb_id].ca

    # instantiate an ANM object
    protein_anm = ANM(protein_ca)

    # build Hessian matrix
    protein_anm.buildHessian(protein_ca)

    protein_anm.calcModes(n_modes, zeros, turbo)

    proteins[pdb_id + '_anm'] = protein_anm

    protein = proteins[pdb_id]
    protein_ca = proteins[pdb_id].ca
    # save ANM modes to the file
    writeNMD(pdb_id+'_anm.nmd', proteins[pdb_id+'_anm'], protein_ca)

    # save atoms
    saveAtoms(protein, pdb_id)

    # save Model
    saveModel(protein_anm, pdb_id+'_ca')

    writeNMD(pdb_id+'_anm.nmd', proteins[pdb_id+'_anm'], proteins[pdb_id].ca)


def anm_extend_model(pdb_id, norm=False):
    """
    Extend existing, coarse grained model built for nodes to atoms.
    This method takes part of the normal modes for each node (Ca atoms)
    and extends it to other atoms in the same residue.

    Creates extended model file (pdb_id+'_ext.nma.npz') for later use.

    :param pdb_id: The four-letter accession code of the desired PDB file
    :type pdb_id: str
    :param norm: If norm is True, extended modes are normalized.
    """
    pdb_id = pdb_id.upper()

    # TODO: Check if needed objects exist in proteins[]
    protein_anm = proteins[pdb_id + '_anm']
    protein_ca = proteins[pdb_id].ca
    protein = proteins[pdb_id]

    protein_ext, protein_all = extendModel(protein_anm, protein_ca, protein, norm)

    proteins[pdb_id+'_anm_ext'] = protein_ext
    proteins[pdb_id+'_all'] = protein_all

    # save extended model
    saveModel(proteins[pdb_id+'_anm_ext'], pdb_id+'_ext')


def sample_conformations(pdb_id, n_confs=1000, rmsd=1.0):
    """
    Sample conformations from along ANM modes.

    :param pdb_id: The four-letter accession code of the desired PDB file
    :param n_confs: Number of conformations to generate
    :param rmsd: average RMSD that the conformations will have with respect to the initial conformation (default: 1.0 Å)
    :return: Ensemble of randomly sampled conformations for the protein's ANM model.
    """
    pdb_id = pdb_id.upper()

    # TODO: Check if needed objects exist in proteins[]
    modes = proteins[pdb_id+'_anm_ext']
    atoms = proteins[pdb_id].protein

    returned_ens = sampleModes(modes, atoms, n_confs, rmsd)

    writeDCD(pdb_id+'_all.dcd', returned_ens)

    return returned_ens


def write_conformations(pdb_id, ensembl):
    pdb_id = pdb_id.upper()

    # TODO: Check if needed objects exist in proteins[]
    protein = proteins[pdb_id]

    protein.addCoordset(ensembl.getCoordsets())

    # set beta values of Ca atoms to 1 and other to 0:
    protein.all.setBetas(0)
    protein.ca.setBetas(1)

    for i in range(1, protein.numCoordsets()):  # skipping 0th coordinate set
        new_dir_path = os.path.join(dir_path, pdb_id.lower()+'_ensemble')

        if not os.path.exists(new_dir_path):  # if needed - create directory for the ensemble
            os.makedirs(new_dir_path)

        fn = os.path.join(new_dir_path, pdb_id.lower() + '_' + str(i) + '.pdb')
        writePDB(fn, protein, csets=i)


def optimize_conformations(pdb_id):
    #  requires: NAMD2 or NAMD3
    if prody.utilities.which('namd3'):
        print("Using NAMD3")
        namd = prody.utilities.which('namd3')

    elif prody.utilities.which('namd2'):
        print("Using NAMD2")
        namd = prody.utilities.which('namd2')

    else:
        print("Couldn't find NAMD2 or NAMD3. Please install before continuing. If it's on your system, make sure  to "
              "add it to PATH")
        return None

    # TODO: Find location of CHARMMPAR file
    # TODO: Do optimization
        ## TODO: Create directory

#############################################################################################


# global variables
dir_path = os.getcwd()

parser = argparse.ArgumentParser(description='')  # TODO: Write description
parser.add_argument('filename', type=str, help='name of PDB file')

args = parser.parse_args()

filename = args.filename
proteins = {}

# TESTING SITE ##########################################################################
sample_load_pdb(filename)
# print(proteins)
# print(proteins.keys())
show_protein(filename)

anm_calc_modes(filename)
print(proteins)
print(proteins[filename.upper()+'_anm'])
showSqFlucts(proteins[filename.upper()+'_anm'])  # (also visible in VMD)
plt.show()
# viewNMDinVMD(filename.upper()+'_anm.nmd')

anm_extend_model(filename)
ens = sample_conformations(filename, 100)

# Begin Analysis
rmsd = ens.getRMSDs()
hist(rmsd, density=False)
xlabel('RMSD')
plt.show()

showProjection(ens, proteins[filename.upper()+'_anm_ext'][:3], rmsd=True)
plt.show()

print(proteins[filename.upper()].numAtoms())
print(ens.numAtoms())

# End Analysis

sample_load_pdb(filename)
write_conformations(filename, ens)
# vmd -m 1p38_ensemble/*pdb


optimize_conformations(filename)
