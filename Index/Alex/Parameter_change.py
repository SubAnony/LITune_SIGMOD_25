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
    new_action.append(action[0])
    new_action.append(action[1])
    if action[2]  == 1:
        new_action.append("true")
    else:
        new_action.append("false")

    if action[3] == 1:
        new_action.append("true")
    else:
        new_action.append("false")

    new_action.append(action[4])
    new_action.append(action[5])

    if action[6] == 1:
        new_action.append("true")
    else:
        new_action.append("false")

    new_action.append(action[7])

    external_kMaxOutOfDomainKeys = action[7] + action[8]


    new_action.append(external_kMaxOutOfDomainKeys)

    new_action.append(action[9])

    new_action.append(min(0.8,action[10]))

    external_kInitDensity = min(0.9,action[10] + action[11])

    new_action.append(external_kInitDensity)

    external_kMaxDensity = min(0.99,external_kInitDensity + action[12])

    new_action.append(external_kMaxDensity)

    new_action.append(action[13])

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
    os.system('echo "#define external_expectedInsertFrac %d" >> %s;'%(action[0],file))
    os.system('echo "#define external_maxNodeSize 1 << %d" >> %s;'%(action[1], file))
    os.system('echo "#define external_approximateModelComputation %s" >> %s;'%(action[2],file))
    os.system('echo "#define external_approximateCostComputation %s" >> %s;'%(action[3],file))
    os.system('echo "#define external_fanoutSelectionMethod %d" >> %s;'%(action[4],file))
    os.system('echo "#define external_splittingPolicyMethod %d" >> %s;'%(action[5],file))
    os.system('echo "#define external_allowSplittingUpwards %s" >> %s;'%(action[6],file))
    os.system('echo "#define external_kMinOutOfDomainKeys %d" >> %s;'%(action[7],file))
    os.system('echo "#define external_kMaxOutOfDomainKeys %d" >> %s;'%(action[8],file))
    os.system('echo "#define external_kOutOfDomainToleranceFactor %d" >> %s;'%(action[9],file))
    os.system('echo "#define external_kMaxDensity %0.4f" >> %s;'%(action[12],file))
    os.system('echo "#define external_kInitDensity %0.4f" >> %s;'%(action[11],file))
    os.system('echo "#define external_kMinDensity %0.4f" >> %s;'%(action[10],file))
    os.system('echo "#define external_kAppendMostlyThreshold %0.4f" >> %s;'%(action[13],file))

    # np.savetxt("./parameters.txt",action)
    with open("./Index/Alex/parameters.txt","w") as f:        
        for item in action:
            if type(item) == np.float64 or type(item) == float or type(item) == np.float32 :
                item = format(item, '0.1f')
            f.write(str(item)+'\n')

    with open("parameters.txt","w") as f:        
        for item in action:
            if type(item) == np.float64 or type(item) == float or type(item) == np.float32 :
                item = format(item, '0.1f')
            f.write(str(item)+'\n')

if __name__ == "__main__":

    # os.chdir("./")
    parser = argparse.ArgumentParser()
    # parser.add_argument("--epsilon", type=int, default=50, help="Epsilon for PGM")
    # parser.add_argument("--ER", default=100, type=int)             
    # args = parser.parse_args()

    action_example = np.array([1,24,1,1,0,0,0,5,995,2,0.7,0,0.1,0.7])
    updateFile("./Index/Alex/src/parameters.hpp",action_example)