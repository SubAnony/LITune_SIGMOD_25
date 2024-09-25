import os
from sre_constants import SUCCESS
import numpy as np
import torch
import gym
import argparse
import os

def action_convert(action):

    """
    1. 0/1 to F/T
    2. Calculate the interval
    
    """   

    new_action = []
    new_action.append(int(2**action[0]))
    new_action.append(int(2**action[1]))
    a2_low_bound = max(new_action[0],new_action[1])+1
    new_action.append(int(a2_low_bound+action[2]))

    new_action.append(action[3])
    new_action.append(action[4])

    new_action.append(action[5])
    new_action.append(action[6])
    new_action.append(action[7])
    new_action.append(action[8])
    new_action.append(action[9])

    new_action.append(action[10])
    new_action.append(action[11])
    new_action.append(action[12])

    if action[13] == 0: 

        new_action.append(-1)
    
    else:
        new_action.append(action[14])

    return new_action

def updateFile(file,raw_action):

    # flags = os.path.exists("./Index/Alex/parameters.txt")
    # if flags:
        # action = np.loadtxt("./Index/Alex/parameters.txt")
        # if action == []:
            # print("Fail to replace parameters")
    # else:
        # action = np.array([1,24,1,1,0,0,0,5,995,2,0.7,0,0.1,0.9])
    action = action_convert(raw_action)

    os.system('> %s;'%file)
    os.system('echo "#pragma once" >> %s;'%file)
    os.system('echo "#define external_kMaxLeafNodeSize %d" >> %s;'%(action[0],file))
    os.system('echo "#define external_kMaxLeafNodeSizeExternal %d" >> %s;'%(action[1], file))
    os.system('echo "#define external_kAlgorithmThreshold %d" >> %s;'%(action[2],file))
    os.system('echo "#define external_kMemoryAccessTime  %0.2f" >> %s;'%(action[3],file))
    os.system('echo "#define external_kLRRootTime  %0.2f" >> %s;'%(action[4],file))
    os.system('echo "#define external_kPLRRootTime %0.2f" >> %s;'%(action[5],file))
    os.system('echo "#define external_kLRInnerTime kMemoryAccessTime + %0.2f" >> %s;'%(action[6],file))
    os.system('echo "#define external_kPLRInnerTime kMemoryAccessTime + %0.2f" >> %s;'%(action[7],file))
    os.system('echo "#define external_kHisInnerTime kMemoryAccessTime + %0.2f" >> %s;'%(action[8],file))
    os.system('echo "#define external_kBSInnerTime kMemoryAccessTime + %0.2f" >> %s;'%(action[9],file))
    os.system('echo "#define external_kCostMoveTime %0.2f" >> %s;'%(action[10],file))
    os.system('echo "#define external_kLeafBaseTime kMemoryAccessTime + %0.1f" >> %s;'%(action[11],file))
    os.system('echo "#define external_kCostBSTime %0.4f" >> %s;'%(action[12],file))
    if action[13] == -1:
        os.system('echo "#define external_lambda %d" >> %s;'%(action[13],file))
    else:
        os.system('echo "#define external_lambda %0.2f" >> %s;'%(action[13],file))

    # np.savetxt("./parameters.txt",action)
    with open("./Index/CARMI/parameters.txt","w") as f:        
        for item in action:
            if type(item) == np.float64 or type(item) == float or type(item) == np.float32 :
                item = format(item, '0.2f')
            f.write(str(item)+'\n')

    with open("parameters.txt","w") as f:        
        for item in action:
            if type(item) == np.float64 or type(item) == float or type(item) == np.float32 :
                item = format(item, '0.2f')
            f.write(str(item)+'\n')

if __name__ == "__main__":

    # os.chdir("./")
    parser = argparse.ArgumentParser()
    # parser.add_argument("--epsilon", type=int, default=50, help="Epsilon for PGM")
    # parser.add_argument("--ER", default=100, type=int)             
    # args = parser.parse_args()

    action_example = np.array([8,10,58975,80.09,11.54,29.62,5.23,22.8,18.44,26.38,6.25,25.4,10.9438,0,1.0])
    updateFile("./Index/Alex/src/parameters.hpp",action_example)