import gym
import numpy as np
from gym import spaces

from scipy.stats import kurtosis
from scipy.interpolate import InterpolatedUnivariateSpline

import cv2

import os

from em_env import EM_Env #Electron microscope interfacing via DMScript marionette

class Fresnel_Env(gym.Env):
    """
    Neural network-electron microscope interface

    This interface has been created to allow a neural network to learn the optimal control 
    policy for removing Fresnel fringes
    """

    def __init__(self, buffer_loc, change_filename, instr_filename, state_filename, state_change_wait, 
                 max_shift, max_z_dist, z_incr, x_bounds, y_bounds, interp_factor = 8):
        self.__version__ = "0.1.0"

        self.buffer_loc = buffer_loc

        #Connect to the electron microscope interface
        self.env = EM_Env(self.buffer_loc+change_filename, self.buffer_loc+instr_filename, 
                          self.buffer_loc+state_filename, state_change_wait)

        #Variables defining the environment
        self.max_shift = max_shift
        self.max_z_dist = max_z_dist
        self.z_incr = z_incr
        self.interp_factor = interp_factor
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

        self.total_shift = (0.0, 0.0)

        #Define the agent's action space
        self._action_set = [self.env.instr_dict["EMSetStageZ"]]
        self.action_space = spaces.Box(low=-self.max_shift, high=self.max_shift, shape=(1,), dtype=np.float32)

        self.get_img_action = self.env.instr_dict["get_img"]

        #Define the observation space
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(self.env.screen_height, self.env.screen_width), 
                                            dtype=np.uint8)
        self.state = self.get_state()
        
        #self.initial_z = self.get_z()

        ##Use calculated correct actions to determine rewards
        #self.req_proximity = 0.1
        #self.z = self.initial_z
        #self.target_z = self.get_optimal_z()

        #self.prev_diff = self.z - self.target_z

    def step(self, action):

        #Act
        self._take_action(action)

        #Observe
        ob = self._get_obs()

        #Get the reward. As the optimal policy is calculated beforehand, this can be calculated from the action
        reward = self._get_reward(action)

        return ob, reward, chain_terminate(), {}

    def _take_action(self, shift):
        '''Apply Z shift and get the resulting image'''
        
        instructions = [[self.action_set[0], shift], 
                        [self.get_img_action]]

        #Store the result of the action as the environment state
        self.state = self.env.execute(instructions)
 
        self.z += shift

    #def soft_reset(self):
    #    '''Repeat the procedure from a random new starting z'''

    #    #move to new location
    #    move_loc()

    #    #Calculate optimum z
    #    get_optimal_z()

    #    #Set z to random starting value
    #    if np.random() > 0.5:
    #        shift_z = np.random()*self.max_z_dist
    #        shift_z(z_shift)
    #    else:
    #        shift_z = -np.random()*self.max_z_dist
    #        shift_z(z_shift)

    def _get_obs(self):
        '''Get image captured from the camera after the action'''
        
        for state in self.state:
            if state[0] == "0":
                self.env.img = self.env.read_img(state[1])

        #self.env.img = self.env.read_img(self.state[0][1])
        return self.env.img

    def _get_reward(shift):
        '''Calculate the reward based on the deviation of the action from the known optimum'''
        
        #Calculate the optimal shift
        displacement = abs(self.target_z - self.z)

        reward = 1.0 if diff <= self.prev_diff else -1.0

        self.prev_diff = 0
        
        return reward

    def chain_terminate(self):
        '''Return whether the chain is within the required proximity'''

        return abs(self.target_z - self.z) < self.req_proximity

    def shift_z(shift):
        '''Shift z by some amount'''

        instructions = [[self.action_st[0], shift],
                        [self.get_img_action]]
        self.state = self.env.execute(instructions)

    def get_state(self):
        '''Get the environment state'''

        return self.env.get_state()

    def terminate(self):
        '''Terminate the interface'''

        self.env.terminate()

    def render(self, mode='None'):
        '''Create an image to display to the user'''

        self.env.render(mode)

    def get_z(self):
        '''Get the current z position'''

        instructions = [[self.env.instr_dict["EMGetStageZ"]]]
        self.env.execute(instructions)

        return float(self.state[0][0])


    @staticmethod
    def fresnel_quantifier(img, rectify=True):
        '''
        Calculates the Kurtosis of an images Laplacian. Low Kurtosis is indicative of the absence
        of Fresnel fringes. By default, the Kurtosis is calculated using positive deviations of
        the Laplacian from its mean
        '''
        flat = cv2.Laplacian(img, cv2.CV_32F, ksize=3).flatten()
        
        if rectify:
            mean = np.mean(flat)

            #Create an array from Laplacians positively deviated from the mean Laplacian
            rectified = np.extract(flat >= mean, flat)
            
            return kurtosis(rectified) #Fisher Kurtosis (-3)  
        else:
            return kurtosis(flat) #Fisher Kurtosis (-3)  

    def go_to_z(self, z):
        '''Go to an absolute z position'''

        instructions = [[self.action_st[0], z],
                        self.get_img_action]
        self.state = self.env.execute(instructions)

    def get_optimal_z(self):
        '''
        Use Kurtosis of Laplaciants to find the height at which Fresnel fringes are removed.
        This is estimated to be at the location of the Kurtosis minimum
        '''

        first_z = self.initial_z-self.max_z_dist
        last_z = self.initial_z+self.max_z_dist
        z_vals = np.linspace(first_z, last_z, self.z_incr)
        kurtosises = np.zeros((len(z_vals,)))
        for i, z in enumerate(z_vals):
            
            self.go_to_z(z)
            kurtosises[i] = self.fresnel_quantifier( self._get_obs()) #Record the kurtosis of the Laplacian at this z

        #Find the location of the minimum using univariate spline interpolation
        ius = InterpolatedUnivariateSpline(z_vals, kurtosises)
        finer_z_vals = np.linspace(first_z, last_z, self.interp_factor*len(kurtosises))
        finer_kurtosises = ius(finer_z_vals)
        
        return finer_z_vals[ finer_kurtosises.argmin() ]

    def reset(self):
        '''Repeat the procedure from a random new starting position and z'''
        
        #Randomly choose a new location
        new_x = self.x_bounds[0] + np.random()*(self.x_bounds[1]-self.x_bounds[0])
        new_y = self.y_bounds[0] + np.random()*(self.y_bounds[1]-self.y_bounds[0])

        shift_x = new_x - self.total_shift[0]
        shift_y = ney_y - self.total_shift[1]

        self.total_shift = (new_x, new_y)

        #Shift to the new position
        instructions = [[self.env.instr_dict["EMChangeBeamShift"], shift_x, shift_y]] #Change beam shift command doesn't work!
        self.state = self.env.execute(instructions)

        #determine the optimal z
        self.target_z = self.get_optimal_z()

        #Set z to random starting value
        shift_z = 0
        if np.random() > 0.5:
            shift_z = np.random()*self.max_z_dist
        else:
            shift_z = -np.random()*self.max_z_dist

        go_to_z(self.target_z+shift_z)
        self.z = self.target_z+shift_z

        self.prev_diff = self.max_z_dist #Reset difference for impending reward calculations

    #def stacks_generator(self, stacks_dir_name=None, stack_size=15):
    #    '''Generate stacks to train the convolutional classifier with'''

    #    if stacks_dir_name == None:
    #        stacks_dir_name = input("Stacks Directory Name: ")

    #    if not os.path.exists(stacks_dir):
    #        os.makedirs(stacks_dir)

    #    stack_num = 0

    #    while(input("Enter 0 to stop? (y/n): ") is not "0"):

    #        stack_num = stack_num+1

    #        x = int(input("X shift: "))
    #        y = int(input("Y shift: "))

    #        instructions = [[self.env.instr_dict["EMSetStageX"], x],
    #                        [self.env.instr_dict["EMSetStageY"], y],
    #                        [self.get_img_action]]
    #        self.state = self.env.execute(instructions)

    #        #determine the optimal z
    #        self.target_z = self.get_optimal_z()

    #        self.
    #        #Generate file to save data to


    #        #Save stack to file

    #        for z in range(self.target_z-self.max_z_dist, self.target_z+self.max_z_dist, stack_size)

    #        go_to_z(self.target_z-shift_z)

    def stacks_generator_focus(self, stacks_dir_name=None, stack_size=15):
        '''Generate stacks to train the convolutional classifier with'''

        if stacks_dir_name == None:
            stacks_dir_name = self.buffer_loc + input("Stacks Directory Name " +self.buffer_loc)

        if not os.path.exists(stacks_dir_name):
            os.makedirs(stacks_dir_name)

        stack_num = 0

        while(input("Enter 0 to stop collecting data: ") is not "0"):

            while(input("Enter 0 to stop changing x and y: ") is not "0"):

                stack_num = stack_num+1
                stack_save_loc = stacks_dir_name+stack_num+"/"

                x = float(input("X shift: "))
                y = float(input("Y shift: "))

                instructions = [[self.env.instr_dict["EMSetStageX"], x],
                                [self.env.instr_dict["EMSetStageY"], y],
                                [self.get_img_action]]
                self.state = self.env.execute(instructions)

                self._get_obs()
                self.env.render("EM_Camera")

            while(input("Enter 0 to stop changing focus: ") is not "0"):

                stack_num = stack_num+1

                df = float(input("Focus change df: "))

                instructions = [[self.env.instr_dict["EMChangeFocus"], df],
                                [self.get_img_action]]
                self.state = self.env.execute(instructions)

                self._get_obs()
                self.env.render("EM_Camera")

            df_range = float(input("Focus range +/-: "))

            #Aquire the image stack
            focus = self.get_focus()
            for i, f in enumerate(np.linspace(focus-df_range, focus+df_range, stack_size)):
                self.set_focus(f)

                #Get and save the image
                self._get_obs()
                cv2.imwrite(stack_save_loc+str(i), self.img)

    def get_focus(self):
        '''Get value of the microscope focus'''

        instructions = [[self.env.instr_dict["EMGetFocus"]]]
        self.state = self.env.execute(instructions)

        return float(self.state[0][1])

    def set_focus(self, f):
        '''Set the microscope focus'''

        instructions = [[self.env.instr_dict["EMSetFocus"], f]]
        self.env.execute(instructions)