import numpy as np
from pycroscopy.io.translators.df_utils.dm_utils import read_dm3, read_dm4

import os, sys
import pickle

from PIL import Image

import time

ALL_FILES_SAVE_LOC = "C:/dump/all_files.P" #Pickle file locations so files only have to be located once
WHITELIST = "C:/dump/whitelist.txt" #Files that can used
DATASET_DIR = "//DESKTOP-SA1EVJV/dataset/" #Location to create dataset
MAX_IMG_SIZE = 4096**2 #Do not save image component of data if it is larger than this
STAGE_LEN = 10_000 #Save data in stages

def shard_dataset():
    """Split dataset into separate files for each tag. This is to speed up loading."""

    #Load all metadata into memory
    metadata = []
    for meta_file in [DATASET_DIR+fn for fn in os.listdir(DATASET_DIR) if "meta-10000.P" in fn]:

        with open(meta_file, 'rb') as f:
            metadata += pickle.load(f)

    #Use availability info to guide tag aquisitions
    with open(DATASET_DIR+"available_data.P", "rb") as f:
        available_data = pickle.load(f)

    for key in available_data:
        data = []
        for i, available in enumerate(available_data[key]):


def record_available_data():
    """Identify tags and record whether or not they are present for each datum."""

    known_keys = []
    for meta_file in [DATASET_DIR+fn for fn in os.listdir(DATASET_DIR) if "meta-10000.P" in fn]:

        with open(meta_file, 'rb') as f:
            metadata = pickle.load(f)

            for dictionary in metadata:
                for key in dictionary:
                    if not key in known_keys:
                        known_keys.append(key)

    with open(DATASET_DIR+"known_keys.P", "wb") as f:
        pickle.dump(known_keys, f)

    available_data = { k: [] for k in known_keys }
    for meta_file in [DATASET_DIR+fn for fn in os.listdir(DATASET_DIR) if "meta-10000.P" in fn]:

        with open(meta_file, 'rb') as f:
            metadata = pickle.load(f)

            for dictionary in metadata:
                for key in known_keys:
                    if key in dictionary:
                        available_data[key].append(True)
                    else:
                        available_data[key].append(False)

    for key in available_data:
        available_data[key] = np.asarray(available_data[key], dtype=np.bool)

    with open(DATASET_DIR+"available_data.P", "wb") as f:
        pickle.dump(available_data, f)


def collect_metadata():
    """Harvest metadata and write it to disk."""

    if os.path.isfile(ALL_FILES_SAVE_LOC):
        #Load list from memory
        with open(ALL_FILES_SAVE_LOC, 'rb') as f:
            files = pickle.load(f)
    else:
        def all_files(dirName):
            """List all files in dir and subdirs routine from 
            https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/"""

            print(dirName) #To give sense of progress..

            listOfFile = os.listdir(dirName)
            allFiles = list()
            # Iterate over all the entries
            for entry in listOfFile:
                # Create full path
                fullPath = os.path.join(dirName, entry)
                # If entry is a directory then get the list of files in this directory 
                if os.path.isdir(fullPath):
                    allFiles = allFiles + all_files(fullPath)
                else:
                    allFiles.append(fullPath)
                
            return allFiles

        #Get parent directories
        with open(WHITELIST, "r") as f:
            parents = [os.path.normcase(l[:-1]) for l in f if l != ""]

        #List files in each parent directory
        files = []
        for p in parents:
            files += [filename for filename in all_files(p) if filename[-4:] in [".dm3", ".dm4"]]

        with open(ALL_FILES_SAVE_LOC, 'wb') as f:
            pickle.dump(files, f)

    meta_list = []
    num_files = len(files)
    for i, filename in enumerate(files[100_001:], 100_001):
        print(f"File {i} of {num_files}...")

        if i and not i % STAGE_LEN:

            #Save metadata to disk
            with open(DATASET_DIR + f"meta-{i}.P", 'wb') as f:
                pickle.dump(meta_list, f)

            #Clear saved data
            meta_list = []

        #Different readers for dm3 and dm4
        reader = read_dm3 if filename[-4:] == ".dm3" else read_dm4
    
        #Get image, metadata and filesystem info
        try:
            img, metadata = reader(filename)
            os_stats = os.stat(filename)
        except:
            continue #Data is corrupt or too difficult to read so discard it

        #Save image if it is smaller than 4096*4096
        if np.product(img.size) <= MAX_IMG_SIZE and "complex" not in str(img.dtype):
        
            metadata.update( {"image_area": np.product(img.size),
                              "image_mean": np.mean(img)} )

            try:
                pass
                #Image.fromarray(img.astype(np.float32)).save( DATASET_DIR + f"images/{i}.tif" )
            except:
                pass #Discard the image


        metadata.update( {"os.stat()": os.stat(filename),
                          "filepath": filename} )

        meta_list.append(metadata)

        ##Save metadata to its own column
        #for meta_key in metadata:
        #    if meta_key in meta_dict:
        #        bool_dict[meta_key].append(True)
        #        meta_dict[meta_key].append(metadata[meta_key])
        #    else:
        #        bool_dict[meta_key] = [False for _ in range((i%STAGE_LEN)+1)]
        #        bool_dict[meta_key][i%STAGE_LEN] = True
        #        meta_dict[meta_key] = [metadata[meta_key]]

        #for meta_key in meta_dict:
        #    if meta_key == "os.stat()":
        #        bool_dict[meta_key].append(True)
        #        meta_dict[meta_key].append(os.stat(filename))
        #    elif meta_key == "filepath":
        #        bool_dict[meta_key].append(True)
        #        meta_dict[meta_key].append(filename)
        #    else:
        #        if meta_key not in metadata:
        #            bool_dict[meta_key].append(False)

if __name__ == "__main__":

    collect_metadata()
    record_available_data()
    shard_dataset()
    