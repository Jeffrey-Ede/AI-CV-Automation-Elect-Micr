import time
import os

import numpy as np

import cv2

class EM_Env_Utility(object):
    '''Utility functions for the EM_Env interface'''

    @staticmethod
    def read_img(img_loc):
        '''Read image at a location'''

        return cv2.imread(img_loc, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def img_to_grey(img):
        '''Average image color channels to convert it to greyscale if it has multiple color channels'''

        return np.mean(img, axis=(0,1))#grey

    def read_grey_img(self, img_loc):
        '''Read and average image color channels to convert it to greyscale if it has multiple color channels'''

        return self.img_to_grey( self.read_img(img_loc) )


class EM_Env(EM_Env_Utility):
    """Electron microscope interface"""

    def __init__(self, change_filename, instr_filename, state_filename, state_change_wait):
        
        #Buffer files
        self.change_filename = change_filename
        self.instr_filename = instr_filename
        self.state_filename = state_filename

        #Check if state has changed after sending instructions every
        self.state_change_wait = state_change_wait #s

        #Environmental variables
        self.screen_width = 672
        self.screen_height = 667

        #Interpret instructions as numbers to send to the microscope
        instr_keys = [
            "get_img", #1 arg: the name to save the image as
            "EMSetStageX", #1 arg: amount to shift the stage X
            "EMSetStageY", #1 arg: amount to shift the stage Y
            "EMSetStageZ", #1 arg: amount to shift the stage Z
            "EMChangeBeamShift", #2 args: shift in x, shift in y
            "terminate"] #0 args
        instr_vals = [str(key) for key in enumerate(instr_keys)]
        self.instr_dict = dict(zip(instr_keys, instr_vals))

        self.img = None

    def execute(self, instructions):
        '''Execute instructions on the electron microscope and get the resulting state'''

        #Pass instructions
        self.write_instr(instructions)
        self.send_instr()

        return get_state()

    def write_instr(self, instructions):
        '''Prepare instructions file for the microscope'''

        with open(self.instr_filename, 'w') as f:
            for instr in instructions:

                #Write the instruction
                f.write(instr[0])

                #Write any argument values
                if len(instr) > 1:
                    for instr_arg in range(1, len(instr)):
                        f.write("\n"+str(instr[instr_arg]))

                #Terminate the instruction chain
                f.write("\n")

    def send_instr(self):
        '''Complete instructions to the microscope by creating a file'''

        with open(self.change_filename, 'w') as f:
            f.write("1")

    def get_state(self):
        '''Check if the state of the electron microscope has changed and interpret its state if it has'''

        #Wait for the state to change
        time.sleep(self.state_change_wait)
        while( self.state_unchanged() ):
            #If state hasn't changed, wait awhile before checking if it has changed again
            time.sleep(self.state_change_wait)

        #Retrieve information about the state
        state_info = []
        with open(self.state_filename, 'r') as f:
            for line in f:
                state_info.append( [x for x in line.split(',')] )

        return state_info

    def state_unchanged(self):
        '''Check if the electron microscope state has changed'''

        return os.path.isfile(self.change_filename)

    def terminate(self):
        '''Free the marionette'''
        
        instruction = [[self.instr_dict["terminate"]]]
        
        self.write_instr(instruction)
        self.send_instr()

    def render(self, mode='None'):
        '''Create an image to display to the user'''
        
        #If the image from the electron microscope camera is requested
        if mode == 'EM_Camera' and self.img is not None:
            
            cv2.imshow('EM_Camera', self.img)
            cv2.waitKey(0)