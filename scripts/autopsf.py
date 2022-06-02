# This script comes from tutorial by Dr. Tamal Banerjee
# THANK YOU

# enter pdb file name
fpdb = "mypdb"
fpsf = "mypsf"

fob = open("runme.tcl", "w")
f1 = fob.write

# enter atomname, atomtype, atomcharge
# in any order, anywhere inside cg
# example: HW1 is renamed in pdb file, HW is the atomtype name I wish to have and 0.464116 is the charge

cg = [['HW1', 'HW', '0.464116'],
      ['HW2', 'HW', '0.464116'],
      ['OW1', 'OW', '-0.928231'],
      ['Cs1', 'CsI', '-0.650000'],
      ['NS1', 'NS', '1.000000'],
      ['OS1', 'OS', '0.950000'],
      ['OS2', 'OS', '-0.650000'],
      ['OS3', 'OS', '-0.650000'],
      ['C', 'C', '0.000000']]

f1("mol new %s.pdb\n\
mol reanalyze top\n\
mol bondsrecalc top\n\
topo guessbonds\n\
topo guessangles\n\
topo guessdihedrals\n\
topo retypebonds\n\n" % (fpdb))

print("Please match the following:")

for i in range(len(cg)):
    var = "s" + str(i)
    f1("set %s [atomselect top {name %s}]\n" % (var, cg[i][0]))
    f1("$%s set type %s\n" % (var, cg[i][1]))
    f1("$%s set charge %s\n\n" % (var, cg[i][2]))
    print("%s %s %s" % (cg[i][0], cg[i][1], cg[i][2]))

f1("animate write psf %s.psf" % fpsf)

print("runme.tcl file generated. Run it in vmd!!")
fob.close()
