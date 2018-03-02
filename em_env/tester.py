import sys
from fresnel_env import Fresnel_Env

buffer_loc = "//flexo.ads.warwick.ac.uk/shared39/EOL2100/2100/Users/Jeffrey-Ede/buffer/"
change_filename = "X.txt"
instr_filename = "instr.txt"
state_filename = "state.txt"
state_change_wait = 0.05 #s

env = Fresnel_Env( buffer_loc, change_filename, instr_filename, state_filename, state_change_wait, 10, 10, 0.5, (-1,1), (-1,1) )
#instructions = [["get_img", "fileLoc"]]

env.stacks_generator_focus()

#env.execute(instructions)
#print(env.get_state())
#env._get_obs()
#print(env.render('EM_Camera'))

env.terminate()