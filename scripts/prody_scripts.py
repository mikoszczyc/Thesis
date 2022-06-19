# Requirements:
# Latest version of ProDy, Matplotlib, VMD and NAMD

from prody import *
from pylab import *
from numpy import *

import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import argparse

proteins = {}
dir_path = os.getcwd()


def load_pdb(pdb_id):
    pdb_id = pdb_id.lower()

    if pdb_id not in proteins:
        fetchPDB(pdb_id)

        # protein = parsePDB(pdb_id)  # , compressed=False
        os.system(f'gunzip {pdb_id}.pdb.gz')
        remove_waters(pdb_id)
        generate_psf(filename)
        protein = parsePDB(pdb_id)  # , compressed=False

        print('ATOMS: ', protein.numAtoms())
        proteins[pdb_id] = protein.protein
        print('PROTEIN ATOMS: ', proteins[pdb_id].numAtoms())

    # Load all types of files that exist (ag, _anm, _ca.anm, _ext.nma)
    if os.path.exists(pdb_id + '.ag.npz'):
        proteins[pdb_id] = loadAtoms(pdb_id + '.ag.npz')

    if os.path.exists(pdb_id + '_ca.anm.npz'):
        proteins[pdb_id + '_anm'] = loadModel(pdb_id + '_ca.anm.npz')

    if os.path.exists(pdb_id + '_ext.nma.npz'):
        proteins[pdb_id + '_anm_ext'] = loadModel(pdb_id + '_ext.nma.npz')


def show_protein(pdb_id):
    pdb_id = pdb_id.lower()
    prody.showProtein(proteins[pdb_id])
    legend()
    plt.show()

def calc_modes(pdb_id, enm='anm', sele='calpha', n_modes=20, zeros=False, turbo=True, cutoff=10.0, gamma=1.0):
    '''
    Perform Elastic Network Model analysis calculations and retrieve normal mode data.
    By default: Creates ANM instance that stores Hessian matrix and normal mode data.
    While using GNM: Creates GNM instance that stores Kirchhoff matrix and normal mode data.
    Normal mode data describes intrinsic dynamics of the protein structure.

    :param pdb_id: The four-letter accession code of the desired PDB file.
    :type pdb_id: string
    :param sele: Selection name.
    :type sele: string
    :param n_modes: Number of non-zero eigenvalues/vectors to calculate. Choose None or 'all' to calculate all modes.
    :type n_modes: int or None, default is 20
    :param zeros: If `True`, modes with zero eigenvalues will be kept.
    :type zeros: bool
    :param turbo: Use faster but more memory intensive calculation mode.
    :type turbo: bool
    :return: ANM / GNM object
    '''

    pdb_id = pdb_id.lower()
    load_pdb(pdb_id)

def extend_model(pdb_id, enm = 'anm'):
    pass

def anm_calc_modes(pdb_id, n_modes=20, zeros=False, turbo=True):  # in tutorial, they used 3 n_modes (default: 20)
    '''
    Perform ANM (Anisotropic Network Model) calculations and retrieve normal mode data.
    Creates ANM instance that stores Hessian matrix and normal mode data.
    Normal mode data describes intrinsic dynamics of the protein structure.

    :param pdb_id: The four-letter accession code of the desired PDB file.
    :type pdb_id: string
    :param n_modes: Number of non-zero eigenvalues/vectors to calculate. Choose None or 'all' to calculate all modes.
    :type n_modes: int or None, default is 20
    :param zeros: If `True`, modes with zero eigenvalues will be kept.
    :type zeros: bool
    :param turbo: Use faster but more memory intensive calculation mode.
    :type turbo: bool
    :return: ANM object
    '''

    pdb_id = pdb_id.lower()
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
    writeNMD(pdb_id + '_anm.nmd', proteins[pdb_id + '_anm'], protein_ca)

    # save atoms
    saveAtoms(protein, pdb_id)

    # save Model
    saveModel(protein_anm, pdb_id + '_ca')

    writeNMD(pdb_id + '_anm.nmd', proteins[pdb_id + '_anm'], proteins[pdb_id].ca)


def anm_extend_model(pdb_id, norm=False):
    '''
    Extend existing, coarse grained model built for nodes to atoms.
    This method takes part of the normal modes for each node (Ca atoms)
    and extends it to other atoms in the same residue.

    Creates extended model file (pdb_id+'_ext.nma.npz') for later use.

    :param pdb_id: The four-letter accession code of the desired PDB file
    :type pdb_id: str
    :param norm: If norm is True, extended modes are normalized.
    '''
    pdb_id = pdb_id.lower()

    # TODO: Check if needed objects exist in proteins[]
    protein_anm = proteins[pdb_id + '_anm']
    protein_ca = proteins[pdb_id].ca
    protein = proteins[pdb_id]

    protein_ext, protein_all = extendModel(protein_anm, protein_ca, protein, norm)

    proteins[pdb_id + '_anm_ext'] = protein_ext
    proteins[pdb_id + '_all'] = protein_all

    # save extended model
    saveModel(proteins[pdb_id + '_anm_ext'], pdb_id + '_ext')


def gnm_calc_modes(pdb_id, selection='calpha', n_modes=20, cutoff=10.0, gamma=1.0, **kwargs):
    pdb_id = pdb_id.lower()

    protein = proteins[pdb_id]
    protein_ca = proteins[pdb_id].ca

    protein_gnm = GNM(pdb_id)
    sele = proteins[pdb_id].select(selection)

    protein_gnm.buildKirchhoff(sele, cutoff, gamma)

    protein_gnm.getKirchhoff()
    protein_gnm.getCutoff()
    protein_gnm.getGamma()

    protein_gnm.calcModes(n_modes, zeros=False, turbo=True)

    protein_gnm.getEigvals().round(3)
    protein_gnm.getEigvecs().round(3)

    slowest_mode = protein_gnm[0]
    slowest_mode.getEigvals().round(3)
    slowest_mode.getEigvecs().round(3)

    protein_gnm.getCovariance().round(2)

    hinges = calcHinges(protein_gnm)
    # hinges = calcHinges(protein_gnm[0])

    proteins[pdb_id + '_gnm'] = protein_gnm
    '''These numbers correspond to node indices in the GNM object, which does not know anything about the original atoms. 
    In order to get the residue numbers corresponding to these hinges, 
    we can index the resum array with the hinges list as follows:'''

    resnums = sele.getResnums()

    mode2_hinges = calcHinges(protein_gnm[1])

    print('Hinge residue numbers: ', resnums[mode2_hinges])

    # Contact Map
    showContactMap(protein_gnm)
    plt.show()

    # Cross-correlations
    showCrossCorr(protein_gnm)
    plt.show()

    # Slow mode shape
    showMode(protein_gnm[0], hinges=True, zero=True)
    grid()
    plt.show()

    # Square fluctuations
    showSqFlucts(protein_gnm[0], hinges=True)
    plt.show()

    # Protein structure bipartition
    showProtein(sele, mode=protein_gnm[0])
    plt.show()

    # save GNM modes to the file
    writeNMD(pdb_id + '_gnm.nmd', proteins[pdb_id + '_gnm'], sele)

    # save atoms
    saveAtoms(protein, pdb_id)

    # save Model
    saveModel(protein_gnm, pdb_id + '_ca')

    # writeNMD(pdb_id + '_gnm.nmd', proteins[pdb_id + '_gnm'], proteins[pdb_id].ca)

    # EXTEND GNM MODEL
    # TODO: Make one method for extending using different modes

    bb_gnm, bb_atoms = extendModel(protein_gnm, sele, protein.select('backbone'))
    print('Backbone GNM: ', bb_gnm.numModes(), bb_gnm.numAtoms())
    print('Backbone atoms: ', bb_atoms.numAtoms())

    proteins[pdb_id + '_gnm_ext'] = bb_gnm
    proteins[pdb_id + '_all'] = bb_atoms

    # save extended model
    saveModel(proteins[pdb_id + '_gnm_ext'], pdb_id + '_ext')

    writeNMD(pdb_id + '_gnm_ext.nmd', bb_gnm[:3], bb_atoms)


def sample_conformations(pdb_id, n_confs=1000, rmsd=1.0):
    '''
    Sample conformations from along ANM modes.

    :param pdb_id: The four-letter accession code of the desired PDB file
    :param n_confs: Number of conformations to generate
    :param rmsd: average RMSD that the conformations will have with respect to the initial conformation (default: 1.0 Å)
    :return: Ensemble of randomly sampled conformations for the protein's ANM model.
    '''
    pdb_id = pdb_id.lower()

    # TODO: Check if needed objects exist in proteins[]
    try:
        modes = proteins[pdb_id + '_anm_ext']
        atoms = proteins[pdb_id].protein

    except:
        print('ANM modes not found! Trying GNM...')
        modes = proteins[pdb_id + '_gnm_ext']
        atoms = proteins[pdb_id + '_all']

    returned_ens = sampleModes(modes, atoms, n_confs, rmsd)

    writeDCD(pdb_id + '_all.dcd', returned_ens)

    return returned_ens


def write_conformations(pdb_id, ensembl):
    pdb_id = pdb_id.lower()

    # TODO: Check if needed objects exist in proteins[]
    protein = proteins[pdb_id]

    protein.addCoordset(ensembl.getCoordsets())

    # set beta values of Ca atoms to 1 and other to 0:
    protein.all.setBetas(0)
    protein.ca.setBetas(1)

    for i in range(1, protein.numCoordsets()):  # skipping 0th coordinate set
        new_dir_path = os.path.join(dir_path, pdb_id.lower() + '_ensemble')

        if not os.path.exists(new_dir_path):  # if needed - create directory for the ensemble
            os.makedirs(new_dir_path)

        fn = os.path.join(new_dir_path, pdb_id.lower() + '_' + str(i) + '.pdb')
        writePDB(fn, protein, csets=i)


def make_namd_conf(pdb_id, timestep=1.0, cutoff=10.0, temperature=0, n_steps=20):
    # TODO: Description
    pdb_id = pdb_id.lower()

    # Check if file exists
    if os.path.isfile(os.path.join(dir_path, 'min.conf')):
        while True:
            val = input('min.conf already exists! Do you want to overwrite it? [Y]/n')
            if val.lower() == 'y' or val.lower() == '':
                break
            elif val.lower() == 'n':
                return None

    # Create min.conf file
    conf_file = open('min.conf', 'w')
    conf_file.write(f'''coordinates\t{{pdb}}
structure       {pdb_id}.psf
paraTypeCharmm  on
parameters      {{par}}
outputname      {{out}}
binaryoutput    no
timestep        {timestep}
cutoff          {cutoff}
switching       on
switchdist      8.0
pairlistdist    12.0
margin          1.0
exclude         scaled1-4
temperature     {temperature}
seed            12345
constraints     on
consref         {{pdb}}
conskfile       {{pdb}}
conskcol        B
constraintScaling  1.0
minimize        {n_steps}
    ''')

    conf_file.close()


def remove_waters(pdb_id):
    # Remove water from pdb
    tcl_cmd = f'''mol load pdb {pdb_id}.pdb
set {pdb_id} [atomselect top protein]
${pdb_id} writepdb {pdb_id}.pdb
exit'''
    with open('remove_waters.tcl', 'w') as inp:
        inp.write(tcl_cmd)
    os.system('vmd -dispdev text -e remove_waters.tcl')


def generate_psf(pdb_id, top='top_all27_prot_lipid_na.inp'):
    pdb_id = pdb_id.lower()

    # Find location of CHARMMPAR file
    tcl_cmd = '''package require readcharmmpar
package require readcharmmtop
global env
set outfile [open charmmdir.txt w]
puts $outfile $env(CHARMMPARDIR)
puts $outfile $env(CHARMMTOPDIR)
close $outfile
exit'''

    with open('where_is_charmmpar.tcl', 'w') as inp:
        inp.write(tcl_cmd)

    os.system('vmd -dispdev text -e where_is_charmmpar.tcl')

    inp = open('charmmdir.txt', 'r')
    lines = inp.readlines()
    inp.close()

    par = os.path.join(lines[0].strip(), 'par_all27_prot_lipid_na.inp')
    top = os.path.join(lines[1].strip(), 'top_all27_prot_lipid_na.inp')
    # if os.path.isfile(os.path.join(dir_path,pdb_id+'.psf')):
    #
    # Generate PSF file
    tcl_cmd = f'''package require psfgen
mol load pdb {pdb_id}.pdb	 
topology {top}	 
pdbalias residue HIS HSE	 
pdbalias atom ILE CD1 CD	 
segment PPP {{pdb {pdb_id}.pdb}}	 
coordpdb {pdb_id}.pdb PPP	 
guesscoord	 
writepdb {pdb_id}.pdb	 
writepsf {pdb_id}.psf
exit'''

    with open('generate_psf.tcl', 'w') as inp:
        inp.write(tcl_cmd)

    os.system('vmd -dispdev text -e generate_psf.tcl > psf.log')
    log_file = open('psf.log', 'r')
    log = log_file.read()

    if log.find('ERROR:') != -1:
        print('Error while generating psf! \n'
              'Check vmd for more info...')
        sys.exit()

    return 0


def optimize_conformations(pdb_id, n_cpu=3, charmm_dir=''):
    import shutil
    pdb_id = pdb_id.lower()
    # TODO: Description

    # CONFIG
    #  requires: NAMD2 or NAMD3
    if prody.utilities.which('namd3'):
        print('Using NAMD3')
        namd = prody.utilities.which('namd3')
        namd_ver = 'namd3'
    elif prody.utilities.which('namd2'):
        print('Using NAMD2')
        namd = prody.utilities.which('namd2')
        namd_ver = 'namd2'
    else:
        print('Couldn\'t find NAMD2 or NAMD3. Please install before continuing. If it\'s on your system, make sure  to '
              'add it to PATH')
        return None

    inp = open('charmmdir.txt', 'r')
    lines = inp.readlines()
    inp.close()

    par = os.path.join(lines[0].strip(), 'par_all27_prot_lipid_na.inp')
    top = os.path.join(lines[1].strip(), 'top_all27_prot_lipid_na.inp')

    # OPTIMIZE
    # Create directory
    new_dir_path = os.path.join(dir_path, pdb_id.lower() + '_optimize')
    if not os.path.exists(new_dir_path):  # if needed - create directory for the optimized ensemble
        print('Making folder for optimization files...')
        os.mkdir(new_dir_path)

    shutil.copyfile(pdb_id + '.psf', pdb_id + '_optimize/' + pdb_id + '.psf')
    # # Generate .psf
    # generate_psf(pdb_id, top)

    # TODO: Split into 2 methods? (NAMD conf files + optimization)
    # Create NAMD configuration file for each conformation based on min.conf file
    import glob
    conf = open('min.conf').read()

    print('Writing NAMD configuration file for each conformation based on min.conf...')
    for pdb in glob.glob(os.path.join(pdb_id + '_ensemble', '*.pdb')):
        fn = os.path.splitext(os.path.split(pdb)[1])[0]
        pdb = os.path.join('..', pdb)
        out = open(os.path.join(pdb_id + '_optimize', fn + '.conf'), 'w')
        out.write(conf.format(out=fn, pdb=pdb, par=par))
        out.close()

    print('Creating folder for optimized data')
    # Optimize conformations
    os.chdir(pdb_id + '_optimize')

    cmds = []  # commands to execute
    for conf in glob.glob('*.conf'):
        fn = os.path.splitext(conf)[0]
        cmds.append(namd_ver + ' ' + conf + ' > ' + fn + '.log')

    from multiprocessing import Pool
    pool = Pool(n_cpu)  # number of CPUs to use

    print(f'Using {n_cpu} CPUs')
    print('Running NAMD')
    signals = pool.map(os.system, cmds)

    if set(signals) == {0}:
        print('NAMD executed correctly')

    # go back to previous dir
    os.chdir('..')


# TODO: parse_coor
# def parse_coor(pdb_id):


# TODO: Work on analysis
def analyze(pdb_id):
    pdb_id = pdb_id.lower()

    initial = AtomGroup(pdb_id + ' initial')
    refined = AtomGroup(pdb_id + ' refined')
    print('Parsing ensembles...')
    for pdb in glob.glob(pdb_id + '_ensemble/*pdb'):
        # print(pdb)
        fn = os.path.splitext(os.path.split(pdb)[1])[0]
        print(fn)

        # rename .coor files to .pdb
        old_name = os.path.join(pdb_id + '_optimize', fn + '.coor')
        new_name = os.path.join(pdb_id + '_optimize', fn + '.pdb')

        if os.path.isfile(old_name):
            os.rename(old_name, new_name)
            opt = new_name
        elif os.path.isfile(new_name):
            opt = new_name
        else:
            continue

        parsePDB(pdb, ag=initial)
        parsePDB(opt, ag=refined)

        # rename back .pdb files to .coor
        old_name = os.path.join(pdb_id + '_optimize', fn + '.coor')
        new_name = os.path.join(pdb_id + '_optimize', fn + '.pdb')

        if os.path.isfile(new_name):
            os.rename(new_name, old_name)

    rmsd_ca = []
    rmsd_all = []
    initial_ca = initial.ca
    refined_ca = refined.ca
    for i in range(initial.numCoordsets()):
        initial.setACSIndex(i)
        refined.setACSIndex(i)
        initial_ca.setACSIndex(i)
        refined_ca.setACSIndex(i)
        rmsd_ca.append(calcRMSD(initial_ca, refined_ca))
        rmsd_all.append(calcRMSD(initial, refined))

    plot(rmsd_all, label='all')
    plot(rmsd_ca, label='ca')
    xlabel('Conformation index')
    ylabel('RMSD')
    legend()
    plt.show()


def analyze_traj(pdb_id):
    pdb_id = pdb_id.lower()
    structure = parsePDB(pdb_id)
    traj = Trajectory(pdb_id + '_all.dcd')

    # Link trajectory to atoms
    traj.link(structure)
    traj.setCoords(structure)

    ensemble = parseDCD(pdb_id + '_all.dcd')
    ensemble.setAtoms(structure)
    ensemble.setCoords(structure)

    ensemble.superpose()

    rmsd = ensemble.getRMSDs()
    print(f'RMSD: {rmsd[:10]}')

    rmsf = ensemble.getRMSFs()
    print(f'RMSF: {rmsf}')

    # Radius of gyration
    protein = structure.select('noh and protein')
    rgyr = zeros(traj.numFrames())

    # TODO: Psi angle

    # Perform calculations
    for i, frame in enumerate(traj):
        rgyr[i] = calcGyradius(protein)

    plot(rgyr)
    xlabel('Frame index')
    ylabel('Radius of gyration (A)')
    plt.show()


#############################################################################################
# global variables


# TESTING SITE ##########################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')  # TODO: Write description
    parser.add_argument('filename', type=str, help='name of PDB file')
    parser.add_argument('enm', choices=['anm', 'gnm'], help='Elastic Network Model')
    args = parser.parse_args()
    filename = args.filename.lower()
    network_model = args.enm.lower()

    load_pdb(filename)
    # show_protein(filename)

    if network_model == 'anm':
        anm_calc_modes(filename, 3)  # def: 20
    elif network_model == 'gnm':
        gnm_calc_modes(filename, 'calpha', 3)

    # showSqFlucts(proteins[filename.lower()+'_anm'])  # (also visible in VMD)
    # plt.show()
    # viewNMDinVMD(filename.lower()+'_anm.nmd')

    if network_model == 'anm':
        anm_extend_model(filename)

    ens = sample_conformations(filename, 100)
    print(ens.numAtoms())
    # Begin Analysis
    rmsd = ens.getRMSDs()
    hist(rmsd, density=False)
    xlabel('RMSD')
    plt.show()
    #
    if network_model == 'anm':
        showProjection(ens, proteins[filename.lower() + '_anm_ext'][:3], rmsd=True)
    else:
        showProjection(ens, proteins[filename.lower() + '_gnm_ext'][:3], rmsd=True)
    plt.show()

    # (proteins[filename.lower()].numAtoms())
    # print(ens.numAtoms())

    # End Analysis
    load_pdb(filename)
    write_conformations(filename, ens)
    # os.system('vmd -m 1p38_ensemble/*pdb')

    make_namd_conf(filename)
    optimize_conformations(filename)
    analyze(filename)
    analyze_traj(filename)
