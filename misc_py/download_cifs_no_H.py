from diffpy.Structure import loadStructure
from diffpy.Structure.expansion import supercell 
from diffpy.structure.symmetryutilities import positionDifference

from CifFile import ReadCif

import numpy as np
import cv2

from PIL import Image

from periodictable import elements

import pickle

import os

from urllib.request import urlopen
from random import shuffle

selection = r"Z:\Jeffrey-Ede\crystal_structures\cod-inorganic\COD-selection.txt"
save_loc = r"Z:\Jeffrey-Ede\crystal_structures\inorganic_no_H\\"

atom_enums = { e.symbol: e.number for e in elements }
atom_enums["D"] = atom_enums["H"]


def process_elem_string(string):
    """Strips ion denotions from names e.g. "O2+" becomes "O"."""

    elem = ""
    for i, c in enumerate(string):
        try:
            int(c)
            break
        except:
            elem += c

    return elem


with open(selection, "r") as f:
    urls = f.read()
    urls = urls.split("\n")
    urls = urls[:-1]

    shuffle(urls)

    temp_filename = f"{save_loc}tmp.cif"
    num_downloaded = 0
    for i, url in enumerate(urls):
        print(f"Iter: {i}")

        try:
            #Download file
            download = urlopen(url).read()

            #Create temporary copy to load structure from
            with open(temp_filename, "wb") as w:
                w.write(download)

            atom_list = loadStructure(temp_filename).tolist()

            #Make sure it doesn't contain hydrogen
            for atom in atom_list:
                elem_num = atom_enums[process_elem_string(atom.element)]
                
                if elem_num == 1:
                    continue

            #Save file
            with open(f"{save_loc}{num_downloaded}.cif", "wb") as w:
                w.write(download)

            num_downloaded += 1

        except:
            pass
