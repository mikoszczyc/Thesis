# Requirements:
# Latest version of ProDy, Matplotlib, VMD and NAMD

import argparse
import glob
import os
import numpy as np

from prody import *
from pylab import *


proteins = {}
dir_path = os.getcwd()


def load_pdb(pdb_id):
    """
    Load PDB file and store in proteins dictionary. If the PDB file is already loaded, do nothing.
    :param pdb_id: The four-letter accession code of the PDB file.
    :type pdb_id: string
    :return: None (stores PDB file in proteins dictionary)
    """

    pdb_id = pdb_id.lower()  # Convert to lowercase to avoid errors.

    if pdb_id not in proteins:  # If the protein is not loaded, load it. If it is loaded, do nothing.
        print('Loading PDB file ' + pdb_id + '.pdb...')
        fetchPDB(pdb_id)  # Fetch PDB file from PDB server. If the PDB file is not available, raise an error.
        os.system(f'gunzip {pdb_id}.pdb.gz')  # Unzip PDB file.
        remove_waters(pdb_id)  # Remove waters.
        generate_psf(filename)  # Generate PSF file. If the PSF file already exists, do nothing.
        protein = parsePDB(pdb_id)  # Parse PDB file. If the PDB file is not available, raise an error.

        print('ATOMS: ', protein.numAtoms())  # print number of atoms in protein. This is useful for debugging. If the number is wrong, the PDB file is probably corrupted.
        proteins[pdb_id] = protein.protein  # Save the protein to the dictionary of proteins. This is useful for later use. The protein is a ProDy object.
        print('PROTEIN ATOMS: ', proteins[pdb_id].numAtoms())  # print number of atoms in protein. This is useful for debugging. This is the same as the number of atoms in the PDB file.


    # TODO: LOADING FILES
    # Load all types of files that exist (ag, _anm, _ca.anm, _ext.nma)
    if os.path.exists(pdb_id + '.ag.npz'):
        proteins[pdb_id] = loadAtoms(pdb_id + '.ag.npz')

    if os.path.exists(pdb_id + '_ca.anm.npz'):
        proteins[pdb_id + '_anm'] = loadModel(pdb_id + '_ca.anm.npz')

    if os.path.exists(pdb_id + '_ext.nma.npz'):
        proteins[pdb_id + '_anm_ext'] = loadModel(pdb_id + '_ext.nma.npz')


def show_protein(pdb_id):
    """
    Show protein structure. Requires ProDy. Requires VMD.

    :param pdb_id: The four-letter accession code of the PDB file.
    :type pdb_id: string
    :return: None (shows protein structure)
    """

    pdb_id = pdb_id.lower()
    if pdb_id not in proteins:  # If the protein is not loaded, load it.
        load_pdb(pdb_id)

    print('Showing protein structure... ' + pdb_id)
    prody.showProtein(proteins[pdb_id])  # Show protein structure.

    legend(loc='upper left')  # Show legend

    plt.show()  # Show plot of the protein.


def calc_modes(pdb_id, sele='calpha', enm='anm', n_modes=20, zeros=False, turbo=True, cutoff=10.0, gamma=1.0):
    """
    Calculate normal modes for a protein using the elastic network model. If the modes already exist, do nothing.

    :param pdb_id: The four-letter accession code of the PDB file.
    :type pdb_id: string
    :param sele: Selection of atoms to use for calculating the normal modes. Default: 'calpha' (Cα atoms).
    :type sele: string
    :param enm: Name of the Elastic Network Model. Default: 'anm' (ANM). Other options: 'gnm' (GNM).
    :type enm: string
    :param n_modes: Number of modes to calculate. Default: 20. If n_modes is None, all modes are calculated.
    :type n_modes: int
    :param zeros: If True, modes with zero eigenvalues will be kept. Default: False.
    :type zeros: bool
    :param turbo: Use a memory intensive, but faster way to calculate modes. Default: True.
    :type turbo: bool
    :param cutoff: Cutoff distance for calculations. Default: 10.0Å (Angstrom).
    :type cutoff: float
    :param gamma: Spring constant. Default: 1.0.
    :type gamma: float
    :return: model, selection (model is the normal modes, selection is the selection of atoms used for calculating the modes) (both are ProDy objects)
    """

    pdb_id = pdb_id.lower()

    load_pdb(pdb_id)  # Load PDB file if it is not already loaded
    protein = proteins[pdb_id]  # Get protein from dictionary.

    selection = protein.select(sele)  # Select the atoms to use for calculating the normal modes (Cα atoms).

    if enm == 'anm':
        model = ANM(protein)  # Create an ANM object.
        print('Building Hessian matrix...')
        model.buildHessian(selection, cutoff=cutoff, gamma=gamma)  # Build the Hessian matrix
    elif enm == 'gnm':
        model = GNM(protein)  # Create a GNM object.
        print('Building Kirchhoff matrix...')
        model.buildKirchhoff(selection, cutoff=cutoff, gamma=gamma)  # Build the Kirchhoff matrix
    else:
        raise ValueError('enm must be either "anm" or "gnm"')  # Raise error if enm is not one of the two options.

    print('Calculating modes...' + pdb_id)
    model.calcModes(n_modes, zeros, turbo)  # Calculate the normal modes (n_modes is the number of modes to calculate)
    print('Saving modes...' + pdb_id)
    saveModel(model)  # Save the normal modes to a file (in .npz format)
    proteins[pdb_id + '_' + enm] = model  # Save the normal modes to the dictionary of proteins.

    if enm == 'gnm':
        # Contact map
        print('Calculating contact map...' + pdb_id)
        showContactMap(model)  # Show the contact map.
        plt.show()

        # Cross-correlations between modes
        print('Calculating cross-correlations...' + pdb_id)
        showCrossCorr(model)  # Show the cross-correlations.
        plt.show()

        # Slow mode shape plot
        print('Calculating slow mode shape plot...' + pdb_id)
        showMode(model[0], hinges=True, zero=True)  # Show the slow mode shape.
        plt.show()

        # Square fluctuations of slow mode
        print('Calculating square fluctuations...' + pdb_id)
        showSqFlucts(model[0], hinges=True)  # Show the square fluctuations of the slow mode shape.
        plt.show()

        # Protein structure bipartition
        print('Calculating protein structure bipartition...' + pdb_id)
        showProtein(selection, mode=model[0]) # Show the protein structure bipartition.
        plt.show()

    # Write NMD file
    print('Writing NMD file...' + pdb_id)
    writeNMD(pdb_id + '_' + enm + '.nmd', model, selection)

    return model, selection


def extend_model(pdb_id, sele='calpha', enm='anm', n_modes=20, zeros=False, turbo=True, cutoff=10.0, gamma=1.0):
    """
    Extend the normal modes of a protein using the elastic network model. If the modes already exist, do nothing. The extended modes are saved to a file.

    :param pdb_id: The four-letter accession code of the PDB file.
    :type pdb_id: string
    :param sele:  Selection of atoms to use for calculating the normal modes. Default: 'calpha' (Cα atoms).
    :type sele: string
    :param enm: Name of the Elastic Network Model. Default: 'anm' (ANM). Other options: 'gnm' (GNM).
    :type enm: string
    :param n_modes: Number of modes to calculate. Default: 20. If n_modes is None, all modes are calculated. If n_modes is -1, the number of modes is determined automatically.
    :type n_modes: int
    :param zeros: If True, modes with zero eigenvalues will be kept. Default: False.
    :type zeros: bool
    :param turbo: Use a memory intensive, but faster way to calculate modes. Default: True.
    :type turbo: bool
    :param cutoff: Cutoff distance for calculations. Default: 10.0Å (Angstrom).
    :type cutoff: float
    :param gamma: Spring constant. Default: 1.0.
    :type gamma: float
    :return: model, selection (model is the extended normal modes, selection is the selection of atoms used for calculating the modes) (both are ProDy objects) (see ProDy documentation for more information - http://prody.csb.cs.nyu.edu/prody/doku.php?id=normal_modes)
    """
    pdb_id = pdb_id.lower()
    load_pdb(pdb_id)

    protein = proteins[pdb_id]

    print('Calculating normal modes...' + pdb_id)
    model, selection = calc_modes(pdb_id, sele=sele, enm=enm, n_modes=n_modes, zeros=zeros, turbo=turbo, cutoff=cutoff,
                                  gamma=gamma)  # Calculate the normal modes

    # Extrapolate to a larger set of atoms
    print('Extrapolating normal modes...' + pdb_id)
    bb_model, bb_atoms = extendModel(model, selection, protein.select('backbone'))  # Extend the normal modes to backbone atoms.
    saveModel(bb_model)  # Save the extended normal modes to a file (in .npz format)

    proteins[pdb_id + '_' + enm + '_ext'] = bb_model  # Save the extended normal modes to the dictionary of proteins.
    proteins[pdb_id + '_all'] = bb_atoms  # Save the extended normal modes to the dictionary of proteins.

    print('Writing NMD file...' + pdb_id)
    writeNMD(pdb_id + '_' + enm + '_ext.nmd', bb_model, bb_atoms)  # Write the extended NMD file.

    return bb_model, bb_atoms   # Return the extended normal modes and the selection of atoms used for calculating the modes.


def sample_conformations(pdb_id, n_confs=1000, rmsd=1.0):
    """
    Sample conformations from along ANM modes. The sampled conformations are saved to a file.

    :param pdb_id: The four-letter accession code of the PDB file.
    :type pdb_id: string
    :param n_confs: Number of conformations to sample. Default: 1000.
    :type n_confs: int
    :param rmsd: average RMSD that the conformations will have with respect to the initial conformation (default: 1.0 Å)
    :type rmsd: float
    :return: Ensemble of sampled conformations (ProDy object).
    """
    pdb_id = pdb_id.lower()

    # Check if needed objects exist in proteins[]. If not, load them. (This is done to avoid loading the same PDB file multiple times.)
    if pdb_id not in proteins:
        print('Loading PDB file...' + pdb_id)
        load_pdb(pdb_id)

    # If selected mode is anm: Get the normal modes and selection of atoms used for calculating the modes. If not, raise error.
    if pdb_id + '_anm' not in proteins:
        raise ValueError('The ANM modes of the protein must be calculated first.')
    else:
        model = proteins[pdb_id + '_anm_ext']
        selection = proteins[pdb_id + '_all']

    # Sample conformations from the normal modes. (The sampled conformations are saved to a file.)
    print('Sampling conformations...' + pdb_id)
    returned_ens = sampleModes(model, selection, n_confs=n_confs, rmsd=rmsd)  # Sample conformations from the normal modes.
    print('Sampled {} conformations.'.format(len(returned_ens)))  # Print the number of sampled conformations.

    saveEnsemble(returned_ens)  # Save the sampled conformations to a file. (in .npz format)
    print('Saved the sampled conformations to a file. (in .npz format)')
    writeDCD(pdb_id + '_all.dcd', returned_ens)  # Write the trajectory to a file. (in .dcd format)
    print('Saved the trajectories to a file. (in .dcd format)')

    return returned_ens  # Return the ensemble of sampled conformations. (ProDy object)


def write_conformations(pdb_id, ensembl):
    """
    Write the sampled conformations to a file.

    :param pdb_id: The four-letter accession code of the PDB file.
    :type pdb_id: string
    :param ensembl: Ensemble of sampled conformations (ProDy object).
    :type ensembl: ProDy object
    :return: None (the sampled conformations are written to a file).
    """

    pdb_id = pdb_id.lower()

    # Check if needed objects exist in proteins[]
    if pdb_id not in proteins:
        raise ValueError('The protein must be loaded first.')

    protein = proteins[pdb_id]  # Get the protein object.

    protein.addCoordset(ensemble.getCoordsets())  # Add the sampled conformations to the protein object.

    # set beta values of Ca atoms to 1 and other to 0:
    protein.all.setBetas(0)  # set beta values of all atoms to 0
    protein.ca.setBetas(1)  # set beta values of Ca atoms to 1

    for i in range(1, protein.numCoordsets()):  # skipping 0th coordinate set (initial conformation)
        new_dir_path = os.path.join(dir_path, pdb_id.lower() + '_ensemble')  # create a new directory for the ensemble.
        if not os.path.exists(new_dir_path):  # create directory for the ensemble (if it doesn't exist)
            os.makedirs(new_dir_path)

        # write the ensemble to a file: (in .pdb format)
        fn = os.path.join(new_dir_path, pdb_id.lower() + '_' + str(i) + '.pdb')
        writePDB(fn, protein, csets=i)
        print('Wrote ensemble to file: ' + fn)  # print the name of the file that was written. (for debugging)


def make_namd_conf(pdb_id, timestep=1.0, cutoff=10.0, temperature=0, n_steps=20):
    """
    Make a NAMD configuration file for the protein. The configuration file is saved to a file.
    The configuration file is used to run NAMD on the protein.

    :param pdb_id: The four-letter accession code of the PDB file.
    :type pdb_id: string
    :param timestep: Timestep for the simulation. Default: 1.0 fs (femtoseconds).
    :type timestep: float
    :param cutoff: Cutoff distance for the LJ interactions. Default: 10.0 Å.
    :type cutoff: float
    :param temperature: Temperature for the simulation. Default: 0 K.
    :type temperature: float
    :param n_steps: Number of steps for the simulation. Default: 20.
    :type n_steps: int
    :return: None (the NAMD configuration file is written to a file).
    """
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
    print('Created min.conf file.')  # print the name of the file that was written. (for debugging)


def remove_waters(pdb_id):
    """
    Remove waters from the protein.
    The waters are removed from the PDB file.

    :param pdb_id:  The four-letter accession code of the PDB file.
    :type pdb_id: string
    :return: None (the waters are removed from the protein object and the waters are removed from the PDB file).
    """

    pdb_id = pdb_id.lower()   # convert to lowercase

    tcl_cmd = f'''mol load pdb {pdb_id}.pdb
set {pdb_id} [atomselect top protein]
${pdb_id} writepdb {pdb_id}.pdb
exit'''  # create tcl command to remove waters from the PDB file. (using atomselect)

    with open('remove_waters.tcl', 'w') as inp:
        inp.write(tcl_cmd)  # write the tcl command to a file.

    print('Removing waters from the PDB file.')
    os.system('vmd -dispdev text -e remove_waters.tcl > remove_waters.log')  # run vmd with the tcl command. (remove waters)

    # Load the PDB file without the waters.
    protein = parsePDB(os.path.join(dir_path, pdb_id + '.pdb'))  # load the PDB file without the waters. (ProDy object)
    proteins[pdb_id] = protein  # add the protein to the proteins dictionary.
    print('Removed waters from the PDB file.')  # print the name of the file that was written. (for debugging)
    print('The protein object has been updated.')

def generate_psf(pdb_id, top='top_all27_prot_lipid_na.inp'):
    """
    Generate a PSF file for the protein.

    :param pdb_id: The four-letter accession code of the PDB file.
    :type pdb_id: string
    :param top: The topology file to use. Default: top_all27_prot_lipid_na.inp
    :type top: string
    :return: None (the PSF file is written to a file). The PSF file is used to run NAMD on the protein.
    """

    pdb_id = pdb_id.lower()  # convert to lowercase

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
        inp.write(tcl_cmd)  # write the tcl command to a file.

    print('Running where_is_charmmpar.tcl to locate CHARMMPAR and CHARMMTOP files...')
    os.system('vmd -dispdev text -e where_is_charmmpar.tcl > where_is_charmmpar.log') # run vmd with the tcl command. (find location of CHARMMPAR file)

    # Read the location of CHARMMPAR file.
    inp = open('charmmdir.txt', 'r')
    lines = inp.readlines()
    inp.close()

    par = os.path.join(lines[0].strip(), 'par_all27_prot_lipid_na.inp')  # get the location of the CHARMMPAR file.
    top = os.path.join(lines[1].strip(), 'top_all27_prot_lipid_na.inp')  # get the location of the CHARMMTOP file.

    # Generate PSF file. (using CHARMMPAR and CHARMMTOP files)
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
        inp.write(tcl_cmd)  # write the tcl command to a file.

    print('Running generate_psf.tcl to generate PSF file...')
    os.system('vmd -dispdev text -e generate_psf.tcl > psf.log')  # run vmd with the tcl command. (generate PSF file)
    log_file = open('psf.log', 'r')  # open the log file.
    log = log_file.read()

    if log.find('ERROR:') != -1:
        print('Error while generating psf! \n'
              'Check log file for more info...')
        sys.exit()  # exit if error while generating psf. (error message is printed in the log file)
    else:
        print('Generated PSF file.')
        log_file.close()

    return 0  # return 0 if no error while generating psf.


def optimize_conformations(pdb_id, n_cpu=3, charmm_dir=''):
    """
    Optimize the conformations of the protein.
    The protein is optimized using NAMD.

    :param pdb_id: The four-letter accession code of the PDB file.
    :type pdb_id: string
    :param n_cpu: The number of CPUs to use. Default: 3
    :type n_cpu: int
    :param charmm_dir: The location of the CHARMM directory. Default: ''
    :type charmm_dir: string
    :return: None (the protein is optimized using NAMD). The optimized protein is written to a file.
    """

    import shutil
    pdb_id = pdb_id.lower()  # convert to lowercase

    # Check which version of NAMD is being used.
    #  requires: NAMD2 or NAMD3
    if prody.utilities.which('namd3'):  # if NAMD3 is installed
        print('Using NAMD3')
        namd = prody.utilities.which('namd3')  # get the location of NAMD3
        namd_ver = 'namd3'
    elif prody.utilities.which('namd2'):  # if NAMD2 is installed
        print('Using NAMD2')
        namd = prody.utilities.which('namd2')  # get the location of NAMD2
        namd_ver = 'namd2'
    else:  # if neither NAMD2 or NAMD3 is installed (error)
        print('Couldn\'t find NAMD2 or NAMD3. Please install before continuing. If it\'s on your system, make sure  to '
              'add it to PATH')  # print error message.
        return None  # exit the function.

    inp = open('charmmdir.txt', 'r')
    lines = inp.readlines()
    inp.close()

    par = os.path.join(lines[0].strip(), 'par_all27_prot_lipid_na.inp')
    top = os.path.join(lines[1].strip(), 'top_all27_prot_lipid_na.inp')

    # OPTIMIZE CONFORMATIONS USING NAMD
    new_dir_path = os.path.join(dir_path, pdb_id.lower() + '_optimize')
    if not os.path.exists(new_dir_path):  # create directory for the optimized ensemble. (if it doesn't exist)
        print('Creating directory for the optimized ensemble...')
        os.mkdir(new_dir_path)
    else:  # if the directory already exists, delete the contents of the directory.
        print('Deleting contents of directory for the optimized ensemble...')
        shutil.rmtree(new_dir_path)
        os.mkdir(new_dir_path)

    shutil.copyfile(pdb_id + '.psf', pdb_id + '_optimize/' + pdb_id + '.psf')  # copy the PSF file to the new directory. (for NAMD)

    # Create NAMD configuration file for each conformation based on min.conf file.
    import glob
    conf = open('min.conf').read()

    print('Writing NAMD configuration file for each conformation based on min.conf...')
    for pdb in glob.glob(os.path.join(pdb_id + '_ensemble', '*.pdb')):  # for each conformation
        fn = os.path.splitext(os.path.split(pdb)[1])[0]  # get the filename without the extension (the conformation number)
        pdb = os.path.join('', pdb)
        out = open(os.path.join(pdb_id + '_optimize', fn + '.conf'), 'w')  # create a new configuration file for the conformation.
        print('Writing NAMD configuration file for conformation ' + fn + '...')
        out.write(conf.format(out=fn, pdb=pdb, par=par))  # write the configuration file.
        out.close()

    print('Creating folder for optimized data')
    # Optimize conformations
    os.chdir(pdb_id + '_optimize')

    cmds = []  # commands to execute
    for conf in glob.glob('*.conf'):  # for each configuration file
        fn = os.path.splitext(conf)[0]  # get the filename without the extension
        cmds.append(namd_ver + ' ' + conf + ' > ' + fn + '.log')  # append the command to the list. (NAMD command)

    from multiprocessing import Pool  # import the multiprocessing module for parallelization.
    pool = Pool(n_cpu)  # number of CPUs to use

    print(f'Using {n_cpu} CPUs')  # print the number of CPUs used. (for debugging)
    print('Running NAMD')
    signals = pool.map(os.system, cmds)  # run the commands in parallel. (NAMD)

    if set(signals) == {0}:  # if all the commands executed successfully. (NAMD)
        print('NAMD executed correctly')  # print message. (for debugging)
    else:  # if not all the commands executed successfully. (NAMD)
        print('NAMD did not execute correctly!')  # print message. (for debugging)

    os.chdir('')  # go back to previous directory.

    return 0  # return 0 if no error while optimizing the conformations.


# TODO: Work on analysis
def analyze(pdb_id):
    """
    Analyze the ensemble.

    :param pdb_id: The four-letter accession code of the PDB file.
    :type pdb_id: string
    :return: None (the ensemble is analyzed).
    """

    pdb_id = pdb_id.lower()  # convert to lowercase

    initial = AtomGroup(pdb_id + ' initial')  # create an AtomGroup object for the initial structure.
    refined = AtomGroup(pdb_id + ' refined')  # create an AtomGroup object for the refined structure.
    print('Parsing ensembles...')
    for pdb in glob.glob(pdb_id + '_ensemble/*pdb'):  # for each conformation in the ensemble (pdb file)
        # print(pdb)
        fn = os.path.splitext(os.path.split(pdb)[1])[0]  # get the filename without the extension (the conformation number)
        print(fn)  # print the conformation number. (for debugging)

        # rename .coor files to .pdb
        if os.path.exists(pdb_id + '_optimize/' + fn + '.coor'):
            os.rename(pdb_id + '_optimize/' + fn + '.coor', pdb_id + '_optimize/' + fn + '.pdb')

        opt = os.path.join(pdb_id + '_optimize', fn + '.pdb')

        parsePDB(pdb, ag=initial)
        parsePDB(opt, ag=refined)

        # rename back .pdb files to .coor
        if os.path.exists(pdb_id + '_optimize/' + fn + '.pdb'):
            os.rename(pdb_id + '_optimize/' + fn + '.pdb', pdb_id + '_optimize/' + fn + '.coor')

    print('Calculating RMSD...')  # calculate the RMSD between the initial and refined structure.
    rmsd_ca = []  # list to store the RMSD between the initial and refined structure. (for CA)
    rmsd_all = []   # list to store the RMSD between the initial and refined structure. (for all atoms)
    initial_ca = initial.ca  # get the CA atoms of the initial structure.
    refined_ca = refined.ca  # get the CA atoms of the refined structure.
    for i in range(initial.numCoordsets()):  # for each conformation in the ensemble
        initial.setACSIndex(i)  # set the conformation number.
        refined.setACSIndex(i)  # set the conformation number.
        initial_ca.setACSIndex(i)
        refined_ca.setACSIndex(i)
        rmsd_ca.append(calcRMSD(initial_ca, refined_ca))  # append the RMSD between the initial and refined structure to the list. (carbon alpha)
        rmsd_all.append(calcRMSD(initial, refined))  # append the RMSD between the initial and refined structure to the list. (all atoms)

    print('RMSD between the initial and refined structure:')
    plot(rmsd_all, label='all')  # plot the RMSD between the initial and refined structure. (all atoms)
    plot(rmsd_ca, label='ca')  # plot the RMSD between the initial and refined structure. (carbon alpha)
    xlabel('Conformation index')
    ylabel('RMSD')
    legend(loc='upper left')  # show the legend.
    plt.show()  # show the plot.

    return 0  # return 0 if no error while analyzing the protein.


def analyze_traj(pdb_id):
    """
    Analyze the ensemble. Trajectory version. Using pyTraj.
    :param pdb_id: The four-letter accession code of the PDB file.
    :type pdb_id: string
    :return: None (the ensemble is analyzed).
    """

    pdb_id = pdb_id.lower()  # convert the pdb_id to lower case.
    print('Parsing ensembles...')

    structure = parsePDB(pdb_id)  # parse the PDB file.
    traj = Trajectory(pdb_id + '_all.dcd')  # create a Trajectory object for the trajectory. (all atoms)

    # Link trajectory to atoms
    traj.link(structure)  # link the trajectory to the structure.
    traj.setCoords(structure)  # set the coordinates of the trajectory to the structure.

    ensemble = parseDCD(pdb_id + '_all.dcd')
    ensemble.setAtoms(structure)
    ensemble.setCoords(structure)

    ensemble.superpose()  # superpose the ensemble on the initial structure.
    # ensemble.write(pdb_id + '_all_superposed.pdb') # write the superposed ensemble to a PDB file.

    rmsd = ensemble.getRMSDs()  # get the RMSD of the ensemble.
    print(f'RMSD: {rmsd[:10]}')  # print the RMSD of the ensemble. (first 10 conformations)

    rmsf = ensemble.getRMSFs()  # get the RMSF of the ensemble.
    print(f'RMSF: {rmsf}')

    # TODO: Radius of gyration

    # TODO: Psi angle

# Example workflow:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')  # TODO: Write description
    parser.add_argument('filename', type=str, help='name of PDB file')
    parser.add_argument('enm', choices=['anm', 'gnm'], help='Elastic Network Model')
    parser.add_argument('--optimize', action='store_true', help='optimize the protein')
    parser.add_argument('--analyze', action='store_true', help='analyze the protein')
    parser.add_argument('--analyze_traj', action='store_true', help='analyze the protein')

    args = parser.parse_args()  # parse the arguments.
    filename = args.filename.lower()  # convert the filename to lower case. (for consistency)
    network_model = args.enm.lower()  # convert the network model to lower case. (for consistency)
    optimize = args.optimize  # get the optimize flag.
    analyze = args.analyze  # get the analyze flag.
    analyze_traj = args.analyze_traj  # get the analyze_traj flag.

    load_pdb(filename)  # load the PDB file. (this is the main function) (for testing)
    show_protein(filename)  # show the protein.

    calc_modes(filename, enm=network_model, n_modes=3)  # calculate the modes.
    # show_modes(filename, n_modes=3)  # TODO: show the modes.
    # show_modes_3d(filename, n_modes=3)  # TODO: show the modes in 3D.
    extend_model(filename, sele='calpha', enm=network_model)  # extend the model.
    # show_extended_model(filename, sele='calpha')  # TODO: show the extended model.

    if input('Do you want to view the model in VMD? ([Y]/n) ') == 'y' or 'yes' or '':
        viewNMDinVMD(filename.lower() + '_' + network_model + '.nmd')

    n_confs = int(input('Enter number of conformations to analyze: [Default: 1000]') or '1000')
    rmsd = float(input('Enter RMSD threshold: [Default: 1.0] ') or '1.0')
    ens = sample_conformations(filename, n_confs=n_confs)  # create ensemble.

    # # Begin Analysis
    # rmsd = ens.getRMSDs()
    # hist(rmsd, density=False)
    # xlabel('RMSD')
    # plt.show()
    # #
    # if network_model == 'anm':
    #     showProjection(ens, proteins[filename.lower() + '_anm_ext'][:3], rmsd=True)
    # plt.show()
    # # End Analysis
    # load_pdb(filename)
    # write_conformations(filename, ens)
    # # os.system('vmd -m 1p38_ensemble/*pdb')

    if optimize:
        make_namd_conf(filename)
        optimize_conformations(filename)  # optimize the protein.

    if analyze:
        analyze(filename)  # analyze the protein.

    if analyze_traj:
        analyze_traj(filename)  # analyze the protein.

# ##########################################################################