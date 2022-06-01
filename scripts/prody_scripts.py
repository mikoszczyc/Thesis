# Requirements:
# Latest version of ProDy, Matplotlib, VMD and NAMD

from prody import *
from pylab import *

import os
import matplotlib
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# from os.path import exists


def sample_load_pdb(pdb_id):  # TODO: Change name
    pdb_id = pdb_id.upper()
    # TODO: Check if file exists

    if pdb_id not in proteins:
        protein = parsePDB(pdb_id)
        proteins[pdb_id] = protein.protein
    # TODO: Save PDB somewhere else
    # TODO: load all types of files that if exist (_anm, _anm_ext, etc)


def show_protein(pdb_id):
    pdb_id = pdb_id.upper()
    prody.showProtein(proteins[pdb_id])
    legend()


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

    modes = proteins[pdb_id+'_anm_ext']
    atoms = proteins[pdb_id].protein
    # atoms = proteins[pdb_id]
    return sampleModes(modes, atoms, n_confs, rmsd)


def write_conformations(pdb_id, ensembl):
    pdb_id = pdb_id.upper()
    protein = proteins[pdb_id]

    protein.addCoordset(ensembl.getCoordsets())

    # set beta values of Ca atoms to 1 and other to 0:
    protein.all.setBetas(0)
    protein.ca.setBetas(1)

    for i in range(1, protein.numCoordsets()):  # skipping 0th coordinate set
        fn = os.path.join('../p38_ensemble', 'p38_' + str(i) + '.pdb')  # TODO: change path
        writePDB(fn, protein, csets=i)

#############################################################################################


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
plt.show()
anm_calc_modes(filename)
print(proteins)
print(proteins[filename.upper()+'_anm'])
showSqFlucts(proteins[filename.upper()+'_anm'])  # (also visible in VMD)
plt.show()
writeNMD(filename.upper()+'_anm.nmd', proteins[filename.upper()+'_anm'], proteins[filename.upper()].ca)
# viewNMDinVMD('1P38_anm.nmd')

anm_extend_model(filename)
ens = sample_conformations(filename, 100)


# writeDCD('p38all.dcd', ens)

rmsd = ens.getRMSDs()
hist(rmsd, density=False)
xlabel('RMSD')
plt.show()

showProjection(ens, proteins[filename.upper()+'_anm_ext'][:3], rmsd=True)
plt.show()

proteins[filename.upper()] = loadAtoms(filename.upper()+'.ag.npz')
proteins[filename.upper()+'_anm'] = loadModel(filename.upper()+'_ca.anm.npz')
proteins[filename.upper()+'_anm_ext'] = loadModel(filename.upper()+'_ext.nma.npz')

print(proteins[filename.upper()].numAtoms())
print(ens.numAtoms())

write_conformations(filename, ens)

# vmd -m p38_ensemble/*pdb