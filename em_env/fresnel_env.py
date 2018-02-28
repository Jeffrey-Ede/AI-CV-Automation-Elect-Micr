import gym
import numpy as np
from gym import spaces

from em_env import EM_Env #Electron microscope interfacing via DMScript marionette

class Fresnel_Env(gym.Env):
    """
    Neural network-electron microscope interface

    This interface has been created to allow a neural network to learn the optimal control 
    policy for removing Fresnel fringes
    """

    def __init__(self, change_filename, instr_filename, state_filename, state_change_wait, max_shift, max_z_dist):
        self.__version__ = "0.1.0"

        #Connect to the electron microscope interface
        self.env = EM_Env( change_filename, instr_filename, state_filename, state_change_wait )

        #Variables defining the environment
        self.max_shift = max_shift
        self.max_z_dist = max_z_dist

        #Define the agent's action space
        self._action_set = [self.env.instr_dict["EMSetStageZ"]]
        self.action_space = spaces.Box(low=-self.max_shift, high=self.max_shift, shape=(1,), dtype=np.float32)

        self.get_img_action = self.env.instr_dict["get_img"]

        #Define the observation space
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(self.env.screen_height, self.env.screen_width), 
                                            dtype=np.uint8)
        self.state = []
        
        #Use calculated correct actions to determine rewards
        self.req_proximity = 1
        self.z = 0
        self.target_z = 0

        self.prev_diff = self.z - self.target_z

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

    def reset(self):
        '''Repeat the procedure from a random new starting position'''

        #move to new location
        move_loc()

        #Calculate optimum z
        get_optimal_z()

        #Set z to random starting value
        if np.random() > 0.5:
            shift_z = np.random()*self.max_z_dist
            shift_z(z_shift)
        else:
            shift_z = -np.random()*self.max_z_dist
            shift_z(z_shift)

    def _get_obs():
        '''Get image captured from the camera after the action'''
        
        return self.env.read_grey_img(self.state[0][1])

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
                        self.get_img_action]
        self.state = self.env.execute(instructions)

    def get_state(self):
        '''Get the environment state'''

        return self.env.get_state()

    def terminate(self):
        '''Terminate the interface'''

        self.env.terminate()
