import os
from shutil import copyfile

from random import shuffle

no_H_dir = r"\\flexo.ads.warwick.ac.uk\shared41\Microscopy\Jeffrey-Ede\crystal_structures\inorganic_no_H\\"
felix_dir = r"\\flexo.ads.warwick.ac.uk\shared41\Microscopy\Jeffrey-Ede\crystal_structures\cod-inorganic\10000_input_files\\"
new_dir = r"\\flexo.ads.warwick.ac.uk\shared41\Microscopy\Jeffrey-Ede\crystal_structures\cifs_no_H\\"

num_new_files = 30_000

no_h_files = [no_H_dir+f for f in os.listdir(no_H_dir)]
felix_dirs = [felix_dir+f+"\\" for f in os.listdir(felix_dir)]

for i in range(num_new_files):
    print(f"Iter: {i}")

    j = i % len(no_h_files)
    k = i % len(felix_dirs)

    if not j:
        shuffle(no_h_files)

    dir = new_dir+f"{i}"
    if not os.path.isdir(dir):
        os.mkdir(dir)

    copyfile(felix_dirs[k] + "felix.inp", new_dir + f"{i}/felix.inp")
    copyfile(felix_dirs[k] + "felix.hkl", new_dir + f"{i}/felix.hkl")
    copyfile(no_h_files[j], new_dir + f"{i}/felix.cif")
