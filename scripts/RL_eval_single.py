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
from hyperopt import fmin,tpe,hp,partial,Trials
from Index.PGM import Parameter_change
from baseline.random_search import random_search
from baseline.grid_search import grid_search
from tqdm import tqdm
from utils import utils
from agents import TD3
from envs.env import LinearFitting, PGMIndex,ALEXIndex,CARMIIndex
from envs.linear_fitting import Linear_model
from agents import TD3, DDPG, dqn, DDPG_Context
from agents import dqn



# def eval_policy(policy, data, search_time,query_type):
#     if args.Index == "ALEX":
#         eval_env = ALEXIndex(data, query_type=query_type)
#     if args.Index == "PGM":
#         eval_env = PGMIndex(data)

#     if args.Index == "CARMI":
#         eval_env = CARMIIndex(data,query_type=query_type)

#     eval_env.reset()
#     state = []
#     best_state = [] 
#     best_runtime= np.inf
#     runtime_list = []
#     avg_reward = 0.
#     start_time = time.time()
#     # search_time = 300 #seconds
#     while (time.time() - start_time) < search_time:
#         state, done = eval_env.reset(), False
#         for _ in range(1):
#             if done:
#                 break
#             raw_action = (policy.select_action(np.array(state)))
#             action = eval_env.action_converter(raw_action)
#             runtime, next_state, reward, done, _ = eval_env.step(action)
#             avg_reward += reward
#             state = next_state
#             runtime_list.append(runtime)
#             if runtime <= best_runtime:
#                 best_runtime = runtime
#                 best_action = action
            
#     print("---------------------------------------")
#     print(f"Evaluation: parameter: {best_action} best runtime:{best_runtime:.5f}")
#     print("---------------------------------------")
#     return [best_action,best_runtime,runtime_list]


def eval_policy(policy, data, search_time, query_type):
    if args.Index == "ALEX":
        eval_env = ALEXIndex(data, query_type=query_type)
    if args.Index == "PGM":
        eval_env = PGMIndex(data)
    if args.Index == "CARMI":
        eval_env = CARMIIndex(data, query_type=query_type)
        
    state = []
    state, done = eval_env.reset(), False
    best_state = [] 
    best_runtime= np.inf
    best_action = None  # Initialize best_action here
    runtime_list = []
    avg_reward = 0.
    start_time = time.time()

    while (time.time() - start_time) < search_time:

        # print("ep+1")
        for _ in range(4):
            if done or (time.time() - start_time) >= search_time:
                break
            
            # print("step+1")
            raw_action = (policy.select_action(np.array(state)))
            action = eval_env.action_converter(raw_action)
            runtime, next_state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            state = next_state
            runtime_list.append(runtime)
            if runtime <= best_runtime:
                best_runtime = runtime
                best_action = action

    if best_action is None:
        print("No action was selected within the time budget.")
    else:
        print("---------------------------------------")
        print(f"Evaluation: parameter: {best_action} best runtime:{best_runtime:.5f}")
        print("---------------------------------------")
    return [best_action, best_runtime, runtime_list]



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--RL_policy", default="DDPG") # Policy name (TD3, DDPG, SAC or DDPG_Context)
    parser.add_argument("--data_file", default='data_0')
    parser.add_argument("--search_budget", default=150, type=int)    # search time budget, seconds
    parser.add_argument("--Index", default='PGM')
    parser.add_argument("--search_method", default='RL', help="method to use")
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", default=False)              # Save model and optimizer parameters
    parser.add_argument("--load_model", default="default")          # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--sample_mode",default="random_local")     # random_local: local + global sampling, random: only global sampling
    parser.add_argument("--query_type", default = 'balanced')       # Set test query types
    
    args = parser.parse_args()

    search_time = args.search_budget

    evaluations = []


    if args.Index == "PGM":

        data_file_name = args.data_file + ".txt"

        env_name = "PGMIndex"

    elif args.Index == "ALEX":

        data_file_name = args.data_file
        env_name = "ALEXIndex"

    elif args.Index == "CARMI":

        data_file_name = args.data_file
        env_name = "CARMIIndex"


    data_name = data_file_name
    query_type = args.query_type

    print("---------------------------------------")
    print(f"Policy: {args.RL_policy}, Env: {env_name}, Seed: {args.seed}")
    print("---------------------------------------")


    if args.Index == "PGM":
        if not os.path.exists(f"./results/{args.search_method}"):
                os.makedirs(f"./results/{args.search_method}")

        env = PGMIndex(data_file_name)

    elif args.Index == "ALEX":

        if not os.path.exists(f"./results/ALEX/{args.search_method}"):
                os.makedirs(f"./results/ALEX/{args.search_method}")

        env = ALEXIndex(data_file_name,query_type=query_type)

    elif args.Index == "CARMI":

        if not os.path.exists(f"./results/CARMI/{args.search_method}"):
            os.makedirs(f"./results/CARMI/{args.search_method}")

        env = CARMIIndex(data_file_name,query_type=query_type)
        
        
    if args.save_model and not os.path.exists("./rlmodels"):
            os.makedirs("./rlmodels")


    # env.reset()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    if args.Index == "PGM":

            state_dim = 2 # type: ignore
            action_dim = env.action_space.n  # type: ignore
            max_action =  env.action_space.n   # type: ignore

    elif args.Index == "ALEX":

            state_dim = 15 # type: ignore
            action_dim = 14  # type: ignore
            max_action =  1   # type: ignore


    elif args.Index == "CARMI":

        state_dim = 15 # type: ignore
        action_dim = 15  # type: ignore
        max_action =  1   # type: ignore
        
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        } 

    # Initialize policy
    if args.RL_policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)

    elif args.RL_policy == "DQN":

        kwargs_dqn = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        } 

        policy = dqn.DQN(**kwargs_dqn)
        
    elif args.RL_policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)
    else:
        policy = DDPG_Context.DDPG(**kwargs)


    
    if args.Index == "PGM":
            file_name = f"{args.RL_policy}_PGMIndex_{args.seed}"
    elif args.Index == "ALEX":
            file_name = f"{args.RL_policy}_ALEXIndex_{args.seed}"
    elif args.Index == "CARMI":
        file_name = f"{args.RL_policy}_CARMIIndex_{args.seed}"

    if args.load_model != "":
            # print("check point 2")
            policy_file = file_name if args.load_model == "default" else args.load_model
            policy.load(f"./rlmodels/{policy_file}")
            # print("check point 3")

    

        # Evaluate untrained policy

    evaluations.append(eval_policy(policy, data_file_name,search_time=search_time,query_type=query_type))
    if args.Index == "PGM":
            np.save(f"./results/{args.search_method}/0_shot_{file_name}_{data_name}", evaluations)
    if args.Index == "ALEX":
            np.save(f"./results/ALEX/{args.search_method}/0_shot_{file_name}_{data_name}_{query_type}", evaluations)
    if args.Index == "CARMI":
            np.save(f"./results/CARMI/{args.search_method}/{file_name}_{data_name}_{query_type}", evaluations)
    if args.save_model: policy.save(f"./rlmodels/{file_name}")


    