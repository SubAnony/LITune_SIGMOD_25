import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import optim
import torchvision.utils
import numpy as np
import random
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gym
from agents import TD3
from baseline import SMBO
from hyperopt import fmin,tpe,hp,partial,Trials,space_eval
import Index.PGM.Parameter_change as pgm_parameter
import Index.Alex.Parameter_change as alex_parameter
import Index.CARMI.Parameter_change as carmi_parameter
from baseline.random_search import random_search
from baseline.grid_search import grid_search
from baseline.heuristic_search import simulated_annealing
import subprocess
from tqdm import tqdm
from utils import utils
from agents import TD3
from envs.env import LinearFitting, PGMIndex
from envs.linear_fitting import Linear_model
from agents import DDPG
import faulthandler


MAX_EVALS = 20
TIME_OUT = 300
# SEARCH_TIME = 300


if __name__ == "__main__":

    faulthandler.enable()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_method", default='random_search', help="method to use")
    parser.add_argument("--Index", default= "PGM")                  
    parser.add_argument("--dynamic", default=False, type=bool)              # dynamic data or static data
    parser.add_argument("--search_budget", default=60, type=int)              # search time budget, seconds
    parser.add_argument("--data_file", default='data_0')# Time steps initial random policy is used
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--query_type", default = 'balanced') #Set test query types

    args = parser.parse_args()


    SEARCH_TIME = args.search_budget

    

    if args.Index == "PGM":

        data_file_name = args.data_file + '.txt'

    elif args.Index == "ALEX":

        data_file_name = args.data_file 

    elif args.Index == "CARMI":

        data_file_name = args.data_file 
        
    query_type = args.query_type

    if args.search_method == "default":

        if args.Index == "PGM":

            pgm_parameter.updateFile("./Index/PGM/index_test.cpp",64,4)
            os.system('g++ ./Index/PGM/index_test.cpp  -w -std=c++17 -o ./Index/PGM/exe_pgm_index')
            os.system(f'./Index/PGM/exe_pgm_index ./data/{data_file_name}')

            f = open("runtime_result.txt",encoding="utf-8")
            cost = float(f.read())
            f.close

            if not os.path.exists(f"./results/{args.search_method}"):
                os.makedirs(f"./results/{args.search_method}")

            file_name = "result"+ f"{args.seed}_{args.data_file}"

            result = []
            result.append(cost)
            print(cost)

            np.save(f"./results/{args.search_method}/{file_name}", result)

        if args.Index == "ALEX":

            action = np.array([1,24,1,1,0,0,0,5,995,2,0.7,0,0.1,0.9])

            alex_parameter.updateFile("./Index/Alex/src/parameters.hpp",action)
            # os.system('sh ./Index/Alex/write_to_parameter.sh')
            os.system('g++ ./Index/Alex/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/Alex/exe_alex_index')
            os.system(f'./Index/Alex/exe_alex_index ./data_SOSD/{data_file_name} {query_type}')

            f = open("runtime_result.txt",encoding="utf-8")
            cost = float(f.read())
            f.close

            if not os.path.exists(f"./results/ALEX/{args.search_method}"):
                os.makedirs(f"./results/ALEX/{args.search_method}")

            file_name = "result"+ f"_{query_type}_{args.seed}_{args.data_file}"

            result = []
            result.append(cost)
            print(cost)

            np.save(f"./results/ALEX/{args.search_method}/{file_name}", result)

        if args.Index == "CARMI":
            
            action = np.array([8,10,58975,80.09,11.54,29.62,5.23,22.8,18.44,26.38,6.25,25.4,10.9438,0,1.0])
            
            carmi_parameter.updateFile("./Index/CARMI/src/parameters.hpp",action)

            os.system('g++ ./Index/CARMI/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/CARMI/exe_carmi_index')
            os.system(f'./Index/CARMI/exe_carmi_index ./data_SOSD/{data_file_name} {query_type}')


            f = open("runtime_result.txt",encoding="utf-8")
            cost = float(f.read())
            f.close

            if not os.path.exists(f"./results/CARMI/{args.search_method}"):
                os.makedirs(f"./results/CARMI/{args.search_method}")

            file_name = "result"+ f"_{query_type}_{args.seed}_{args.data_file}"

            result = []
            result.append(cost)
            print(cost)

            np.save(f"./results/CARMI/{args.search_method}/{file_name}", result)


    if args.search_method == "BO":

        random.seed(args.seed)

        def hyperopt_model_score(params):
            if args.Index == "PGM":
                pgm_parameter.updateFile("./Index/PGM/index_test.cpp", params['epsilon'], params['ER'])
                os.system('g++ ./Index/PGM/index_test.cpp  -w -std=c++17 -o ./Index/PGM/exe_pgm_index')
                cmd = f'./Index/PGM/exe_pgm_index ./data/{data_file_name}'

            elif args.Index == "ALEX":
                params_list = [params['external_expectedInsertFrac'],
                            params['external_maxNodeSize_factor'],
                            params['external_approximateModelComputation'],
                            params['external_approximateCostComputation'],
                            params['external_fanoutSelectionMethod'],
                            params['external_splittingPolicyMethod'],
                            params['external_allowSplittingUpwards'],
                            params['external_kMinOutOfDomainKeys'],
                            params['external_kinterval'],
                            params['external_kOutOfDomainToleranceFactor'],
                            params['external_kMinDensity'],
                            params['external_kDensityinterval_1'],
                            params['external_kDensityinterval_2'],
                            params['external_kAppendMostlyThreshold']]

                alex_parameter.updateFile("./Index/Alex/src/parameters.hpp", params_list)
                os.system('g++ ./Index/Alex/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/Alex/exe_alex_index')
                cmd = f'./Index/Alex/exe_alex_index ./data_SOSD/{data_file_name} {query_type}'

            elif args.Index == "CARMI":

                params_list = [  params['kMaxLeafNodeSize'],
                                params['kMaxLeafNodeSizeExternal'],
                                params['kAlgorithmThreshold'],
                                params['kMemoryAccessTime'],
                                params['kLRRootTime'],
                                params['kPLRRootTime'],
                                params['kLRInnerTime'],
                                params['kPLRInnerTime'],
                                params['kHisInnerTime'],
                                params['kBSInnerTime'],
                                params['kCostMoveTime'],
                                params['kLeafBaseTime'],
                                params['kCostBSTime'],
                                params['external_lambda_int'],
                                params['external_lambda_float']]
                
                carmi_parameter.updateFile("./Index/CARMI/src/parameters.hpp",params_list)

                os.system('g++ ./Index/CARMI/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/CARMI/exe_carmi_index')
                cmd = f'./Index/CARMI/exe_carmi_index ./data_SOSD/{data_file_name} {query_type}'

            else:
                print("Please define the correct Index")
                return 0

            process = None
            try:
                process = subprocess.Popen(cmd, shell=True)
                process.wait(timeout=TIME_OUT)
            except subprocess.TimeoutExpired:
                # Handle the timeout case here
                if process is not None:
                    # Try to terminate the process gracefully
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        # If process is still running after 1 second, kill it with SIGKILL
                        process.kill()
                    
                return 9999 

            f = open("runtime_result.txt", encoding="utf-8")
            cost = float(f.read())
            f.close()

            return cost



        def f_model(params):
            acc = hyperopt_model_score(params)
            return acc

        if args.Index == "PGM":

            param_grid = {
            'epsilon': hp.choice('epsilon', range(1,8000) ),
            'ER': hp.choice('ER', range(1,20))}

        elif args.Index == "ALEX":

            # action = np.array([1,24,1,1,0,0,0,5,995,2,0.7,0,0.1,0.9])
            param_grid = {
            'external_expectedInsertFrac': hp.uniform('external_expectedInsertFrac', 0,1),
            'external_maxNodeSize_factor': hp.choice('external_maxNodeSize_factor',range(20,30)),
            'external_approximateModelComputation': hp.choice('external_approximateModelComputation',[0,1]),
            'external_approximateCostComputation': hp.choice('external_approximateCostComputation',[0,1]),
            'external_fanoutSelectionMethod':hp.choice('external_fanoutSelectionMethod',[0]),
            'external_splittingPolicyMethod':hp.choice('external_splittingPolicyMethod',[0]),
            'external_allowSplittingUpwards':hp.choice('external_allowSplittingUpwards',[0]),
            'external_kMinOutOfDomainKeys':hp.choice('external_kMinOutOfDomainKeys',range(0,500,1)),
            'external_kinterval':hp.choice('external_kinterval',range(1000,5000,1)),
            'external_kOutOfDomainToleranceFactor':hp.choice('external_kOutOfDomainToleranceFactor',range(1,51)),
            'external_kMinDensity':hp.uniform('external_kMinDensity',0,1),
            'external_kDensityinterval_1':hp.uniform('external_kDensityinterval_1',0,0.5),
            'external_kDensityinterval_2':hp.uniform('external_kDensityinterval_2',0,0.5),
            "external_kAppendMostlyThreshold":hp.uniform('external_kAppendMostlyThreshold',0,1)}


            #reset parameters
            action = np.array([1,24,1,1,0,0,0,5,995,2,0.7,0,0.1,0.9])

            alex_parameter.updateFile("./Index/Alex/src/parameters.hpp",action)
            os.system('g++ ./Index/Alex/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/Alex/exe_alex_index')
            os.system(f'./Index/Alex/exe_alex_index ./data_SOSD/{data_file_name} {query_type}')
    
        elif args.Index == "CARMI":

                param_grid = {
                'kMaxLeafNodeSize': hp.choice('kMaxLeafNodeSize',range(5,15)),
                'kMaxLeafNodeSizeExternal':hp.choice('kMaxLeafNodeSizeExternal',range(5,15)),
                'kAlgorithmThreshold':hp.choice('kAlgorithmThreshold',range(1,300000,10)),
                'kMemoryAccessTime':hp.uniform('kMemoryAccessTime',1,5e2),
                'kLRRootTime':hp.uniform('kLRRootTime',1,5e2),
                'kPLRRootTime': hp.uniform('kPLRRootTime',1,5e2),
                'kLRInnerTime':hp.uniform('kLRInnerTime',1,5e2),
                'kPLRInnerTime':hp.uniform('kPLRInnerTime',1,5e2),
                'kHisInnerTime':hp.uniform('kHisInnerTime',1,5e2),
                'kBSInnerTime':hp.uniform('kBSInnerTime',1,5e2),
                'kCostMoveTime':hp.uniform('kCostMoveTime',1,5e2),
                'kLeafBaseTime':hp.uniform('kLeafBaseTime',1,5e2),
                'kCostBSTime': hp.uniform('kCostBSTime',1,5e2),
                'external_lambda_int':hp.choice('external_lambda_int',[0,1]),
                'external_lambda_float':hp.uniform('external_lambda_float',0.1,100)
                }


                #reset parameters

                action = np.array([8,10,58975,80.09,11.54,29.62,5.23,22.8,18.44,26.38,6.25,25.4,10.9438,0,1.0])
                
                carmi_parameter.updateFile("./Index/CARMI/src/parameters.hpp",action)

                os.system('g++ ./Index/CARMI/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/CARMI/exe_carmi_index')
                os.system(f'./Index/CARMI/exe_carmi_index ./data_SOSD/{data_file_name} {query_type}')


        trials = Trials()
        start = time.monotonic()
        best = fmin(f_model, param_grid, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials,timeout=SEARCH_TIME)
        end = time.monotonic()
        best_hyp = space_eval(param_grid, best)
        time_tuning = end- start
        print('best:')
        print(best_hyp)
        print('time used:')
        print(time_tuning)
        print('best runtime')
        print(hyperopt_model_score(best_hyp))

        if args.Index == "PGM":

            if not os.path.exists(f"./results/{args.search_method}"):
                os.makedirs(f"./results/{args.search_method}")

            file_name = "result"+ f"_{args.seed}_{args.data_file}"

            result = []
            result.append(hyperopt_model_score(best_hyp))
            result.append(time_tuning)
            result.append(best_hyp)

            np.save(f"./results/{args.search_method}/{file_name}", result)

        if args.Index == "ALEX":

            if not os.path.exists(f"./results/ALEX/{args.search_method}"):
                os.makedirs(f"./results/ALEX/{args.search_method}")

            file_name = "result"+ f"_{query_type}_{args.seed}_{args.data_file}"

            result = []
            result.append(hyperopt_model_score(best_hyp))
            result.append(time_tuning)
            result.append(best_hyp)

            np.save(f"./results/ALEX/{args.search_method}/{file_name}", result)

        if args.Index == "CARMI":

            if not os.path.exists(f"./results/CARMI/{args.search_method}"):
                os.makedirs(f"./results/CARMI/{args.search_method}")

            file_name = "result"+ f"_{query_type}_{args.seed}_{args.data_file}"

            result = []
            result.append(hyperopt_model_score(best_hyp))
            result.append(time_tuning)
            result.append(best_hyp)

            np.save(f"./results/CARMI/{args.search_method}/{file_name}", result)

    if args.search_method == "heuristic_search":
        random.seed(args.seed)
        best_score = np.inf
        best_hyperparams = []

        # Add the rest of your original code here to initialize the param_grid and other variables

        search_model = simulated_annealing

        initial_temperature = 100
        cooling_rate = 0.99
        max_iterations = MAX_EVALS


        if args.Index == "PGM":

            param_grid = {
            'epsilon': hp.choice('epsilon', range(1,8000) ),
            'ER': hp.choice('ER', range(1,20))}

        elif args.Index == "ALEX":

            # action = np.array([1,24,1,1,0,0,0,5,995,2,0.7,0,0.1,0.9])
            param_grid = {
            'external_expectedInsertFrac': list(np.arange(0,1,0.001)),
            'external_maxNodeSize_factor': list(range(20,30)),
            'external_approximateModelComputation': [0,1],
            'external_approximateCostComputation': [0,1],
            'external_fanoutSelectionMethod':[0],
            'external_splittingPolicyMethod':[0],
            'external_allowSplittingUpwards':[0],
            'external_kMinOutOfDomainKeys':list(range(2,500,1)),
            'external_kinterval':list(range(1000,5000,1)),
            'external_kOutOfDomainToleranceFactor':list(range(1,51)),
            'external_kMinDensity':list(np.arange(0,1,0.001)),
            'external_kDensityinterval_1':list(np.arange(0,0.5,0.001)),
            'external_kDensityinterval_2':list(np.arange(0,0.5,0.001)),
            "external_kAppendMostlyThreshold":list(np.arange(0,1,0.001))}


            #reset parameters
            action = np.array([1,24,1,1,0,0,0,5,995,2,0.7,0,0.1,0.9])

            alex_parameter.updateFile("./Index/Alex/src/parameters.hpp",action)
            os.system('g++ ./Index/Alex/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/Alex/exe_alex_index')
            os.system(f'./Index/Alex/exe_alex_index ./data_SOSD/{data_file_name} {query_type}')

        elif args.Index == "CARMI":

            param_grid = {
            'kMaxLeafNodeSize': list(range(5,15)),
            'kMaxLeafNodeSizeExternal':list(range(5,15)),
            'kAlgorithmThreshold':list(range(1,300000,100)),
            'kMemoryAccessTime':list(np.arange(1,5e2,0.1)),
            'kLRRootTime':list(np.arange(1,5e2,0.1)),
            'kPLRRootTime': list(np.arange(1,5e2,0.1)),
            'kLRInnerTime':list(np.arange(1,5e2,0.1)),
            'kPLRInnerTime':list(np.arange(1,5e2,0.1)),
            'kHisInnerTime':list(np.arange(1,5e2,0.1)),
            'kBSInnerTime':list(np.arange(1,5e2,0.1)),
            'kCostMoveTime':list(np.arange(1,5e2,0.1)),
            'kLeafBaseTime':list(np.arange(1,5e2,0.1)),
            'kCostBSTime': list(np.arange(1,5e2,0.1)),
            'external_lambda_int':[0,1],
            'external_lambda_float':list(np.arange(0.1,100,0.01))
            }

            action = np.array([8,10,58975,80.09,11.54,29.62,5.23,22.8,18.44,26.38,6.25,25.4,10.9438,0,1.0])
            
            carmi_parameter.updateFile("./Index/CARMI/src/parameters.hpp",action)

            os.system('g++ ./Index/CARMI/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/CARMI/exe_carmi_index')
            os.system(f'./Index/CARMI/exe_carmi_index ./data_SOSD/{data_file_name} {query_type}')

        def model_loss(params):
    
            if args.Index == "PGM":
                pgm_parameter.updateFile("./Index/PGM/index_test.cpp", params[0], params[1])
                os.system('g++ ./Index/PGM/index_test.cpp  -w -std=c++17 -o ./Index/PGM/exe_pgm_index')
                cmd = f'./Index/PGM/exe_pgm_index ./data/{data_file_name} {query_type}'

            # other Index in progress 

            elif args.Index == "ALEX":
                alex_parameter.updateFile("./Index/Alex/src/parameters.hpp", params)
                os.system('g++ ./Index/Alex/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/Alex/exe_alex_index')
                cmd = f'./Index/Alex/exe_alex_index ./data_SOSD/{data_file_name} {query_type}'

            elif args.Index == "CARMI":

                carmi_parameter.updateFile("./Index/CARMI/src/parameters.hpp",params)

                os.system('g++ ./Index/CARMI/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/CARMI/exe_carmi_index')
                cmd = f'./Index/CARMI/exe_carmi_index ./data_SOSD/{data_file_name} {query_type}'


            process = None
            try:
                process = subprocess.Popen(cmd, shell=True)
                process.wait(timeout=TIME_OUT)
            except subprocess.TimeoutExpired:
                # Handle the timeout case here
                if process is not None:
                    # Try to terminate the process gracefully
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        # If process is still running after 1 second, kill it with SIGKILL
                        process.kill()
                    
                return 9999 
            
            f = open("runtime_result.txt",encoding="utf-8")
            cost = float(f.read())
            f.close()
            
            return cost 

        start = time.monotonic()

        best_hyperparams, best_score = search_model(param_grid=param_grid,
                                                    initial_temperature=initial_temperature,
                                                    cooling_rate=cooling_rate,
                                                    max_iterations=max_iterations,
                                                    model_loss=model_loss,
                                                    time_budget=SEARCH_TIME)

        end = time.monotonic()
        time_tuning = end - start

        print('best:')
        print(best_hyperparams)
        print('time used:')
        print(time_tuning)
        print('best runtime')
        print(best_score)


        if args.Index == "PGM":

            if not os.path.exists(f"./results/{args.search_method}"):
                os.makedirs(f"./results/{args.search_method}")

            file_name = "result"+ f"_{args.data_file}"

            result = []
            result.append(model_loss(best_hyperparams[0],best_hyperparams[1]))
            result.append(time_tuning)
            result.append(best_hyperparams)

            np.save(f"./results/{args.search_method}/{file_name}", result)

        elif args.Index == "ALEX":

            if not os.path.exists(f"./results/ALEX/{args.search_method}"):
                os.makedirs(f"./results/ALEX/{args.search_method}")

            file_name = "result"+ f"_{query_type}_{args.seed}_{args.data_file}"

            result = []
            result.append(best_score)
            result.append(time_tuning)
            result.append(best_hyperparams)

            np.save(f"./results/ALEX/{args.search_method}/{file_name}", result)

        elif args.Index == "CARMI":

            if not os.path.exists(f"./results/CARMI/{args.search_method}"):
                os.makedirs(f"./results/CARMI/{args.search_method}")


            file_name = "result"+ f"_{query_type}_{args.seed}_{args.data_file}"

            result = []
            result.append(best_score)
            result.append(time_tuning)
            result.append(best_hyperparams)

            np.save(f"./results/CARMI/{args.search_method}/{file_name}", result)

        
    if args.search_method == "random_search":

        random.seed(args.seed)
        # 记录用
        best_score = np.inf
        best_hyperparams = []

        if args.Index == "PGM":

            param_grid = {
            'epsilon': list(np.arange(1,8000)),
            'b': list(np.arange(1,20))
            }

        elif args.Index == "ALEX":

            param_grid = {
            'external_expectedInsertFrac': list(np.arange(0,1,0.001)),
            'external_maxNodeSize_factor': list(range(20,30)),
            'external_approximateModelComputation': [0,1],
            'external_approximateCostComputation': [0,1],
            'external_fanoutSelectionMethod':[0],
            'external_splittingPolicyMethod':[0],
            'external_allowSplittingUpwards':[0],
            'external_kMinOutOfDomainKeys':list(range(0,500,1)),
            'external_kinterval':list(range(1000,5000,1)),
            'external_kOutOfDomainToleranceFactor':list(range(1,51)),
            'external_kMinDensity':list(np.arange(0,1,0.001)),
            'external_kDensityinterval_1':list(np.arange(0,0.5,0.001)),
            'external_kDensityinterval_2':list(np.arange(0,0.5,0.001)),
            "external_kAppendMostlyThreshold":list(np.arange(0,1,0.001))}

            #reset parameters
            action = np.array([1,24,1,1,0,0,0,5,995,2,0.7,0,0.1,0.9])

            alex_parameter.updateFile("./Index/Alex/src/parameters.hpp",action)
            os.system('g++ ./Index/Alex/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/Alex/exe_alex_index')
            os.system(f'./Index/Alex/exe_alex_index ./data_SOSD/{data_file_name} {query_type}')

        elif args.Index == "CARMI":

            param_grid = {
            'kMaxLeafNodeSize': list(range(5,15)),
            'kMaxLeafNodeSizeExternal':list(range(5,15)),
            'kAlgorithmThreshold':list(range(1,300000,100)),
            'kMemoryAccessTime':list(np.arange(1,5e2,0.1)),
            'kLRRootTime':list(np.arange(1,5e2,0.1)),
            'kPLRRootTime': list(np.arange(1,5e2,0.1)),
            'kLRInnerTime':list(np.arange(1,5e2,0.1)),
            'kPLRInnerTime':list(np.arange(1,5e2,0.1)),
            'kHisInnerTime':list(np.arange(1,5e2,0.1)),
            'kBSInnerTime':list(np.arange(1,5e2,0.1)),
            'kCostMoveTime':list(np.arange(1,5e2,0.1)),
            'kLeafBaseTime':list(np.arange(1,5e2,0.1)),
            'kCostBSTime': list(np.arange(1,5e2,0.1)),
            'external_lambda_int':[0,1],
            'external_lambda_float':list(np.arange(0.1,100,0.01))
            }

            action = np.array([8,10,58975,80.09,11.54,29.62,5.23,22.8,18.44,26.38,6.25,25.4,10.9438,0,1.0])
            
            carmi_parameter.updateFile("./Index/CARMI/src/parameters.hpp",action)

            os.system('g++ ./Index/CARMI/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/CARMI/exe_carmi_index')
            os.system(f'./Index/CARMI/exe_carmi_index ./data_SOSD/{data_file_name} {query_type}')




        def model_loss(params):
    
            if args.Index == "PGM":
                pgm_parameter.updateFile("./Index/PGM/index_test.cpp", params[0], params[1])
                os.system('g++ ./Index/PGM/index_test.cpp  -w -std=c++17 -o ./Index/PGM/exe_pgm_index')
                cmd = f'./Index/PGM/exe_pgm_index ./data/{data_file_name} {query_type}'

            # other Index in progress 

            elif args.Index == "ALEX":
                alex_parameter.updateFile("./Index/Alex/src/parameters.hpp", params)
                os.system('g++ ./Index/Alex/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/Alex/exe_alex_index')
                cmd = f'./Index/Alex/exe_alex_index ./data_SOSD/{data_file_name} {query_type}'

            elif args.Index == "CARMI":

                carmi_parameter.updateFile("./Index/CARMI/src/parameters.hpp",params)

                os.system('g++ ./Index/CARMI/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/CARMI/exe_carmi_index')
                cmd = f'./Index/CARMI/exe_carmi_index ./data_SOSD/{data_file_name} {query_type}'

            process = None
            try:
                process = subprocess.Popen(cmd, shell=True)
                process.wait(timeout=TIME_OUT)
            except subprocess.TimeoutExpired:
                # Handle the timeout case here
                if process is not None:
                    # Try to terminate the process gracefully
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        # If process is still running after 1 second, kill it with SIGKILL
                        process.kill()
                    
                return 9999 
            
            f = open("runtime_result.txt",encoding="utf-8")
            cost = float(f.read())
            f.close()
            
            return cost 
        
        random.seed(random.randint(1,10))

        search_model = random_search()
        start = time.monotonic()

        # for i in tqdm(range(MAX_EVALS)):

        start_time = time.time()

        i = 0

        while (time.time() - start_time) < SEARCH_TIME:

    
            params = search_model.random_search(param_grid=param_grid,discrete = True)


            score = model_loss(params)
            print(f"---random searching on {i}th iteration, run time is {score} s, parameter is {params}")
            i = i+1
            if score < best_score:
                best_hyperparams = params
                best_score = score
            # torch.save(model.state_dict(), "best_model.pt")

        end = time.monotonic()
        time_tuning = end- start
        print('best:')
        print(best_hyperparams)
        print('time used:')
        print(time_tuning)
        print('best runtime')
        print(best_score)

        if args.Index == "PGM":

            if not os.path.exists(f"./results/{args.search_method}"):
                os.makedirs(f"./results/{args.search_method}")

            file_name = "result"+ f"_{args.data_file}"

            result = []
            result.append(model_loss(best_hyperparams[0],best_hyperparams[1]))
            result.append(time_tuning)
            result.append(best_hyperparams)

            np.save(f"./results/{args.search_method}/{file_name}", result)

        elif args.Index == "ALEX":

            if not os.path.exists(f"./results/ALEX/{args.search_method}"):
                os.makedirs(f"./results/ALEX/{args.search_method}")

            file_name = "result"+ f"_{query_type}_{args.seed}_{args.data_file}"

            result = []
            result.append(best_score)
            result.append(time_tuning)
            result.append(best_hyperparams)

            np.save(f"./results/ALEX/{args.search_method}/{file_name}", result)

        elif args.Index == "CARMI":

            if not os.path.exists(f"./results/CARMI/{args.search_method}"):
                os.makedirs(f"./results/CARMI/{args.search_method}")


            file_name = "result"+ f"_{query_type}_{args.seed}_{args.data_file}"

            result = []
            result.append(best_score)
            result.append(time_tuning)
            result.append(best_hyperparams)

            np.save(f"./results/CARMI/{args.search_method}/{file_name}", result)


        

    if args.search_method == "grid_search":

        random.seed(args.seed)
    
        best_score = np.inf
        best_hyperparams = []


        if args.Index == "PGM":

            param_grid = {
            'epsilon': list(np.arange(1,8000)),
            'b': list(np.arange(1,20))
            }

        elif args.Index == "ALEX":

            param_grid = {
            'external_expectedInsertFrac': list(np.arange(0.1,1,0.2)),
            'external_maxNodeSize_factor': list(range(20,25)),
            'external_approximateModelComputation': [0,1],
            'external_approximateCostComputation': [0,1],
            'external_fanoutSelectionMethod':[0],
            'external_splittingPolicyMethod':[0],
            'external_allowSplittingUpwards':[0],
            'external_kMinOutOfDomainKeys':list(range(0,500,100)),
            'external_kinterval':list(range(1000,5000,1000)),
            'external_kOutOfDomainToleranceFactor':list(range(1,51,10)),
            'external_kMinDensity':list(np.arange(0.2,0.8,0.1)),
            'external_kDensityinterval_1':list(np.arange(0,0.3,0.1)),
            'external_kDensityinterval_2':list(np.arange(0,0.3,0.1)),
            "external_kAppendMostlyThreshold":list(np.arange(0.4,0.9,0.1))}

            #Alternative parameter grid
            # param_grid = {
            # 'external_expectedInsertFrac': list(np.arange(0,1,0.001)),
            # 'external_maxNodeSize_factor': list(range(20,30)),
            # 'external_approximateModelComputation': [0,1],
            # 'external_approximateCostComputation': [0,1],
            # 'external_fanoutSelectionMethod':[0],
            # 'external_splittingPolicyMethod':[0],
            # 'external_allowSplittingUpwards':[0],
            # 'external_kMinOutOfDomainKeys':list(range(0,500,1)),
            # 'external_kinterval':list(range(1000,5000,1)),
            # 'external_kOutOfDomainToleranceFactor':list(range(1,51)),
            # 'external_kMinDensity':list(np.arange(0,1,0.001)),
            # 'external_kDensityinterval_1':list(np.arange(0,0.5,0.001)),
            # 'external_kDensityinterval_2':list(np.arange(0,0.5,0.001)),
            # "external_kAppendMostlyThreshold":list(np.arange(0,1,0.001))}
            #reset parameters
            action = np.array([1,24,1,1,0,0,0,5,995,2,0.7,0,0.1,0.9])

            alex_parameter.updateFile("./Index/Alex/src/parameters.hpp",action)
            os.system('g++ ./Index/Alex/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/Alex/exe_alex_index')
            os.system(f'./Index/Alex/exe_alex_index ./data_SOSD/{data_file_name} {query_type}')

        elif args.Index == "CARMI":

            param_grid = {
            'kMaxLeafNodeSize': list(range(5,15,3)),
            'kMaxLeafNodeSizeExternal':list(range(5,15,3)),
            'kAlgorithmThreshold':list(range(1,300000,50000)),
            'kMemoryAccessTime':list(np.arange(1,5e2,200)),
            'kLRRootTime':list(np.arange(1,5e2,200)),
            'kPLRRootTime': list(np.arange(1,5e2,200)),
            'kLRInnerTime':list(np.arange(1,5e2,300)),
            'kPLRInnerTime':list(np.arange(1,5e2,300)),
            'kHisInnerTime':list(np.arange(1,5e2,300)),
            'kBSInnerTime':list(np.arange(1,5e2,300)),
            'kCostMoveTime':list(np.arange(1,5e2,300)),
            'kLeafBaseTime':list(np.arange(1,5e2,300)),
            'kCostBSTime': list(np.arange(1,5e2,300)),
            'external_lambda_int':[0,1],
            'external_lambda_float':list(np.arange(0.1,100,20))
            }
            
            #Alternative parameter grid
            # param_grid = {
            # 'kMaxLeafNodeSize': list(range(5,15)),
            # 'kMaxLeafNodeSizeExternal':list(range(5,15)),
            # 'kAlgorithmThreshold':list(range(1,300000,100)),
            # 'kMemoryAccessTime':list(np.arange(1,5e2,0.1)),
            # 'kLRRootTime':list(np.arange(1,5e2,0.1)),
            # 'kPLRRootTime': list(np.arange(1,5e2,0.1)),
            # 'kLRInnerTime':list(np.arange(1,5e2,0.1)),
            # 'kPLRInnerTime':list(np.arange(1,5e2,0.1)),
            # 'kHisInnerTime':list(np.arange(1,5e2,0.1)),
            # 'kBSInnerTime':list(np.arange(1,5e2,0.1)),
            # 'kCostMoveTime':list(np.arange(1,5e2,0.1)),
            # 'kLeafBaseTime':list(np.arange(1,5e2,0.1)),
            # 'kCostBSTime': list(np.arange(1,5e2,0.1)),
            # 'external_lambda_int':[0,1],
            # 'external_lambda_float':list(np.arange(0.1,100,0.01))
            # }


            action = np.array([8,10,58975,80.09,11.54,29.62,5.23,22.8,18.44,26.38,6.25,25.4,10.9438,0,1.0])
            
            carmi_parameter.updateFile("./Index/CARMI/src/parameters.hpp",action)

            os.system('g++ ./Index/CARMI/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/CARMI/exe_carmi_index')
            os.system(f'./Index/CARMI/exe_carmi_index ./data_SOSD/{data_file_name} {query_type}')

        result =[]

        def model_loss(params):
    
            if args.Index == "PGM":
                pgm_parameter.updateFile("./Index/PGM/index_test.cpp", params[0], params[1])
                os.system('g++ ./Index/PGM/index_test.cpp  -w -std=c++17 -o ./Index/PGM/exe_pgm_index')
                cmd = f'./Index/PGM/exe_pgm_index ./data/{data_file_name} {query_type}'

            # other Index in progress 

            elif args.Index == "ALEX":
                alex_parameter.updateFile("./Index/Alex/src/parameters.hpp", params)
                os.system('g++ ./Index/Alex/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/Alex/exe_alex_index')
                cmd = f'./Index/Alex/exe_alex_index ./data_SOSD/{data_file_name} {query_type}'

            elif args.Index == "CARMI":

                carmi_parameter.updateFile("./Index/CARMI/src/parameters.hpp",params)

                os.system('g++ ./Index/CARMI/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/CARMI/exe_carmi_index')
                cmd = f'./Index/CARMI/exe_carmi_index ./data_SOSD/{data_file_name} {query_type}'

            process = None
            try:
                process = subprocess.Popen(cmd, shell=True)
                process.wait(timeout=TIME_OUT)
            except subprocess.TimeoutExpired:
                # Handle the timeout case here
                if process is not None:
                    # Try to terminate the process gracefully
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        # If process is still running after 1 second, kill it with SIGKILL
                        process.kill()
                    
                return 9999 
            
            f = open("runtime_result.txt",encoding="utf-8")
            cost = float(f.read())
            f.close()
            
            return cost 


        search_model = grid_search()

        step = 0
        hp_set = search_model.grid_search(param_grid=param_grid)
        start = time.monotonic()

        def flatten(data):
            if isinstance(data, tuple):
                for x in data:
                    yield from flatten(x)
            else:
                yield data

        start_time = time.time()

        for params in tqdm(hp_set):
            params = list(flatten(params))
            # print(params)

            step += 1

            if (time.time() - start_time) > SEARCH_TIME:
                break
            score = model_loss(params)

            print(f"---grid searching on {step}th iteration, run time is {score} s")
            if score < best_score:
                best_hyperparams = params
                best_score = score
            # torch.save(model.state_dict(), "best_model.pt")

        end = time.monotonic()
        time_tuning = end- start
        print('best:')
        print(best_hyperparams)
        print('time used:')
        print(time_tuning)
        print('best runtime')
        print(best_score)


        if args.Index == "PGM":

            if not os.path.exists(f"./results/{args.search_method}"):
                os.makedirs(f"./results/{args.search_method}")

            file_name = "result"+ f"_{args.data_file}"

            result = []
            result.append(model_loss(best_hyperparams[0],best_hyperparams[1]))
            result.append(time_tuning)
            result.append(best_hyperparams)

            np.save(f"./results/{args.search_method}/{file_name}", result)

        elif args.Index == "ALEX":

            if not os.path.exists(f"./results/ALEX/{args.search_method}"):
                os.makedirs(f"./results/ALEX/{args.search_method}")

            file_name = "result"+ f"_{query_type}_{args.seed}_{args.data_file}"

            result = []
            result.append(best_score)
            result.append(time_tuning)
            result.append(best_hyperparams)

            np.save(f"./results/ALEX/{args.search_method}/{file_name}", result)


        elif args.Index == "CARMI":

            if not os.path.exists(f"./results/CARMI/{args.search_method}"):
                os.makedirs(f"./results/CARMI/{args.search_method}")


            file_name = "result"+ f"_{query_type}_{args.seed}_{args.data_file}"

            result = []
            result.append(best_score)
            result.append(time_tuning)
            result.append(best_hyperparams)

            np.save(f"./results/CARMI/{args.search_method}/{file_name}", result)




        



        


        

        
        


