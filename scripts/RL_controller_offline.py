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

def eval_policy(policy, data, query_type, eval_episodes=4):
    if args.Index == "ALEX":
        eval_env = ALEXIndex(data,query_type=query_type)
    if args.Index == "PGM":
        eval_env = PGMIndex(data)
    if args.Index == "CARMI":
        eval_env = CARMIIndex(data,query_type=query_type)


    eval_env.reset()
    state = []
    best_state = [] 
    best_runtime= np.inf
    runtime_list = []
    avg_reward = 0.
    for i in range(eval_episodes):
        state, done = eval_env.reset(), False
        eval_env.seed(100+i)
        for j in range(5):
            raw_action = (policy.select_action(np.array(state)))
            action = eval_env.action_converter(raw_action)
            runtime, next_state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            state = next_state
            runtime_list.append(runtime)
            if runtime <= best_runtime:
                best_runtime = runtime
                best_action = action
            
            

    best_runtime_real = min(runtime_list)
    avg_reward = avg_reward/(eval_episodes*5)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.5f} parameter: {best_action} best runtime:{best_runtime:.5f}")
    print("---------------------------------------")
    return avg_reward



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--RL_policy", default="DDPG") # Policy name (TD3, DDPG, SAC or DDPG_Context)
    parser.add_argument("--data_file", default='data_0')
    parser.add_argument("--Index", default='ALEX')
    parser.add_argument("--search_method", default='RL', help="method to use")
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=50, type=int)# Time steps initial random policy is used
    parser.add_argument("--max_timesteps", default=1200, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", default=False)              # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--sample_mode",default="random_local")     # random_local: local + global sampling, random: only global sampling
    parser.add_argument("--env_change_freq",default=100)     # random_local: local + global sampling, random: only global sampling
    parser.add_argument("--query_type", default = 'balanced') #Set test query types
    parser.add_argument("--mode", default ='Initialization') # Initialization, training, validatioon (working mode in transfer learning)

    # Env Related
    parser.add_argument("--use-terminal-action", type=bool, default=True,
                        help="whether to use terminal action wrapper")
    parser.add_argument("--use-reward-difference", type=bool, default=True,
                        help="whether to use difference in reward")
    parser.add_argument("--reward-scaling", type=str, default='linear',
                        help="use cubic/exponential scaling to encourage risky behaviour")
    parser.add_argument("--denoise-threshold", type=int, default=-1,
                        help="denoise small rewards for numerical stability")
    parser.add_argument("--episode-timesteps", type=int, default=200,
                        help="hard limit on timesteps per episode")
    parser.add_argument("--action-history", type=int, default=40,
                        help="number of history actions to record")
    parser.add_argument("--encoding", type=str, default='ir2vec',
                        help="encoding for bitcode programs")
    parser.add_argument("--record-rollouts", type=str, default="ppo_rollouts",
                        help="location to save rollouts")


    args = parser.parse_args()

    if args.Index == "PGM":

        data_file_name = args.data_file + ".txt"

        env_name = "PGMIndex"

    elif args.Index == "ALEX":

        data_file_name = args.data_file
        env_name = "ALEXIndex"

    elif args.Index == "CARMI":

        data_file_name = args.data_file
        env_name = "CARMIIndex"


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

    env.reset()
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
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./rlmodels/{policy_file}")
        print("load model success")

    
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    

    state, done = env.reset(), False
    episode_runtime = 0
    episode_timesteps = 0
    episode_num = 0
    episode_reward_list = []
    episode_runtime_list = []
    lowest_reward = 0
    episode_reward = 0
    accumulated_score = []

    MAX_EPI_STEPS = 100

    change_i = 0

    print("---------Training on 0th data set---------")

    for t in range(int(args.max_timesteps)):


        if (t + 1) % args.env_change_freq ==0  and change_i <10:
            
            if args.mode == "training":
                change_i +=10
                
            elif args.mode == "validate":
                
                change_i +=20
                
            else:
                
                continue

            change_i += 1
            data_name = "data_" + f"{change_i}"
            
            print(f"---------Training on {change_i}th data set---------")

            if args.Index == "PGM":
                data_file_name = data_name+".txt"
                env = PGMIndex(data_file_name)
            if args.Index == "ALEX":
                data_file_name = data_name
                env = ALEXIndex(data_file_name, query_type=query_type)
            if args.Index == "CARMI":
                data_file_name = data_name
                env = CARMIIndex(data_file_name, query_type=query_type)


        episode_timesteps += 1

        # Select action randomly or according to policy
        if args.RL_policy != "DQN":
            if t < args.start_timesteps:
                raw_action = env.action_space.sample()

                if args.Index == "ALEX":

                    #pre-process the sampled actions
                    action = [raw_action['external_expectedInsertFrac'],raw_action['external_maxNodeSize_factor'],
                    raw_action['external_approximateModelComputation'],raw_action['external_approximateCostComputation'],raw_action['external_fanoutSelectionMethod'],
                    raw_action['external_splittingPolicyMethod'],raw_action['external_allowSplittingUpwards'],raw_action['external_kMinOutOfDomainKeys'],raw_action['external_kinterval'],
                    raw_action['external_kOutOfDomainToleranceFactor'],raw_action['external_kMinDensity'],raw_action['external_kDensityinterval_1'],raw_action['external_kDensityinterval_2'],
                    raw_action['external_kAppendMostlyThreshold']]
                    action[0] = float(action[0])
                    action[1] = env.external_maxNodeSize_factor_list[int(action[1])]
                    action[7] = env.external_kMinOutOfDomainKeys_list[int(action[7])]
                    action[8] = env.kinterval_list[int(action[8])]
                    action[9] = env.external_kOutOfDomainToleranceFactor_list[int(action[9])]
                    action[10] = float(action[10])
                    action[11] = float(action[11])
                    action[12] = float(action[12])
                    action[13] = float(action[13])
                elif args.Index == "CARMI":

                    action = [raw_action['kMaxLeafNodeSize'],raw_action['kMaxLeafNodeSizeExternal'],raw_action['kAlgorithmThreshold'],
                              raw_action['kMemoryAccessTime'],raw_action['kLRRootTime'],raw_action['kPLRRootTime'],raw_action['kLRInnerTime'],
                              raw_action['kPLRInnerTime'],raw_action['kHisInnerTime'],raw_action['kBSInnerTime'],raw_action['kCostMoveTime'],
                              raw_action['kLeafBaseTime'],raw_action['kCostBSTime'],raw_action['external_lambda_int'],raw_action['external_lambda_float']
                              ]
                    
                    action[0] = env.external_kMaxLeafNodeSize_factor_list[int(action[0])]
                    action[1] = env.external_kMaxLeafNodeSizeExternal_factor_list[int(action[1])]
                    action[2] = int(action[2])
                    action[3] = float(action[3])
                    action[4] = float(action[4])
                    action[5] = float(action[5])
                    action[6] = float(action[6])
                    action[7] = float(action[7])
                    action[8] = float(action[8])
                    action[9] = float(action[9])
                    action[10] = float(action[10])
                    action[11] = float(action[11])
                    action[12] = float(action[12])
                    action[13] = int(action[13])
                    action[14] = float(action[14])


            else:
                raw_action = (policy.select_action(np.array(state),add_noise=False)+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(0,max_action)
                action = env.action_converter(raw_action)

                

            # Perform action
            # new_action = env.action_converter(action)
            runtime, next_state, reward, done, _ = env.step(action) 
            done_bool = float(done) if episode_timesteps < MAX_EPI_STEPS else 0

            # Store data in replay buffer
            # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward
            episode_runtime_list.append(runtime)

        # Train agent after collecting sufficient data
        
            if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)  # type: ignore
        else:

            if t < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(state))

                # take action
            next_state, reward, done, _ = env.step(action)
            policy.store_transition(state, action, reward,next_state)  # type: ignore
            state = next_state
            episode_reward += reward
            episode_reward_list.append(reward)
            
            if t >= args.start_timesteps:
                if policy.memory_counter > 20:  # type: ignore
                    policy.train()  # type: ignore

        
        if  (episode_timesteps) % MAX_EPI_STEPS == 0 or done: #To do: Set stop criterion

            if args.RL_policy != "DQN":
                best_runtime = min(episode_runtime_list)
            else:
                best_reward = max(episode_reward_list)
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.5f}, Best runtime: {best_runtime:.5f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            episode_reward_list = []
            episode_runtime_list = []

        
        
        if args.save_model: policy.save(f"./rlmodels/{file_name}")
        
        if args.mode == 'validate':
            score = eval_policy(policy,data_file_name,query_type)
            
        accumulated_score.append(score)
        
    if args.mode == 'validate':
        
        print("VALIDATION_SCORE:", accumulated_score)





