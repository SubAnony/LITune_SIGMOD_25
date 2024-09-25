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


def eval_policy(policy, data, query_type, search_time=150):
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
    start_time = time.time()
    ep = 0
    # search_time = 150 #seconds
    while (time.time() - start_time) < search_time:
        state, done = eval_env.reset(), False
        eval_env.seed(100+random.randint(1,10))
        ep += 1
        for _ in range(16):
            if done:
                break
            raw_action = (policy.select_action(np.array(state)))
            action = eval_env.action_converter(raw_action)
            runtime, next_state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            state = next_state
            runtime_list.append(runtime)
            if runtime <= best_runtime:
                best_runtime = runtime
                best_action = action
            
    epsodic_reward = avg_reward / ep
    print("---------------------------------------")
    print(f"Evaluation: parameter: {best_action} best runtime:{best_runtime:.5f}")
    print("---------------------------------------")
    return [epsodic_reward, best_action,best_runtime,runtime_list]




class O2System:
    def __init__(self, update_interval, threshold):
        self.online_model = PretrainedRLModel()
        self.offline_model = OfflineRLModel()
        self.update_interval = update_interval
        self.threshold = threshold

    def check_data_divergence(self):
        # For simplicity, let's simulate divergence with a random float
        return random.random()

    def run(self):
        while True:
            time.sleep(self.update_interval)

            divergence = self.check_data_divergence()
            if divergence > self.threshold:
                # Update the offline model
                self.offline_model.refine()

                # Query offline model for an update decision
                if self.offline_model.should_update(self.online_model):
                    self.online_model.update_from(self.offline_model)

            else:
                self.online_model.optimize()
                
        
class PretrainedRLModel:
    def __init__(self,policy_file,args):
        
        data_file_name = args.data_file
        query_type = args.query_type


        if args.Index == "PGM":
            if not os.path.exists(f"./results/{args.search_method}"):
                os.makedirs(f"./results/{args.search_method}")

            self.env = PGMIndex(data_file_name)

        elif args.Index == "ALEX":

            if not os.path.exists(f"./results/ALEX/{args.search_method}"):
                os.makedirs(f"./results/ALEX/{args.search_method}")

            self.env = ALEXIndex(data_file_name,query_type=query_type)

        elif args.Index == "CARMI":

            if not os.path.exists(f"./results/CARMI/{args.search_method}"):
                os.makedirs(f"./results/CARMI/{args.search_method}")

            self.env = CARMIIndex(data_file_name,query_type=query_type)

        
        if args.save_model and not os.path.exists("./rlmodels"):
            os.makedirs("./rlmodels")

        self.env.reset()
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
        if args.RL_policy == "DDPG":
            self.model_online = DDPG.DDPG(**kwargs)
        else:
            self.model_online = DDPG_Context.DDPG(**kwargs)
        self.model_online.load(f"./rlmodels/{policy_file}")
        # self.optimized_data=args.online_data
        print("Online model initialized with pre-trained data.")
        
    def load_model(self,policy_file):
            
        self.model_online.load(f"./rlmodels/{policy_file}")

    def optimize(self):
        print("Online model optimizing in real-time.")

    def update_from(self, offline_model):
        print("Online model updated based on the offline model.")

class OfflineRLModel:
    def __init__(self,policy_file,args):
    
    
        data_file_name = args.data_file
        query_type = args.query_type


        if args.Index == "PGM":
            if not os.path.exists(f"./results/{args.search_method}"):
                os.makedirs(f"./results/{args.search_method}")

            self.env = PGMIndex(data_file_name)

        elif args.Index == "ALEX":

            if not os.path.exists(f"./results/ALEX/{args.search_method}"):
                os.makedirs(f"./results/ALEX/{args.search_method}")

            self.env = ALEXIndex(data_file_name,query_type=query_type)

        elif args.Index == "CARMI":

            if not os.path.exists(f"./results/CARMI/{args.search_method}"):
                os.makedirs(f"./results/CARMI/{args.search_method}")

            self.env = CARMIIndex(data_file_name,query_type=query_type)

        
        if args.save_model and not os.path.exists("./rlmodels"):
            os.makedirs("./rlmodels")

        self.env.reset()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)


        if args.Index == "PGM":

            self.state_dim = 2 # type: ignore
            self.action_dim = env.action_space.n  # type: ignore
            self.max_action =  env.action_space.n   # type: ignore

        elif args.Index == "ALEX":

            self.state_dim = 15 # type: ignore
            self.action_dim = 14  # type: ignore
            self.max_action =  1   # type: ignore

        elif args.Index == "CARMI":

            self.state_dim = 15 # type: ignore
            self.action_dim = 15  # type: ignore
            self.max_action =  1   # type: ignore
        kwargs = {
        "state_dim": self.state_dim,
        "action_dim": self.action_dim,
        "max_action": self.max_action,
        "discount": args.discount,
        "tau": args.tau,
        } 
        if args.RL_policy == "DDPG":
            self.model_offline = DDPG.DDPG(**kwargs)
        else:
            self.model_offline = DDPG_Context.DDPG(**kwargs)
        
        self.model_offline.load(f"./rlmodels/{policy_file}")
        
        self.replay_buffer = utils.ReplayBuffer(self.state_dim, self.action_dim)

        # Evaluate untrained policy
        # self.evaluations = [eval_policy(_online, data_file_name, query_type= query_type, search_time=search_time)]

        # self.state, self.done = env.reset(), False
        # self.episode_runtime = 0
        # self.episode_timesteps = 0
        # self.episode_num = 0
        # self.episode_reward_list = []
        # self.episode_runtime_list = []
        # self.lowest_reward = 0
        # self.episode_reward = 0


        # self.change_i = 0

        self.data_name = "data_11"
        print("Offline model initialized.")
        
    def load_model(self,policy_file):
        
        self.model_offline.load(f"./rlmodels/{policy_file}")
        
    def save_model(self, file_name):
        self.model_offline.save(f"./rlmodels/{file_name}")
        

    def refine(self,refined_data,args):
        print("Offline model refining based on changing data.")
        
        #Compare online model with offline model
        query_type = args.query_type
        if args.Index == "PGM":
            env = PGMIndex(refined_data)
            state, done = env.reset(), False
        if args.Index == "ALEX":
            env = ALEXIndex(refined_data,query_type=query_type)
            state, done = env.reset(), False
        if args.Index == "CARMI":
            env = CARMIIndex(refined_data,query_type=query_type)
            state, done = env.reset(), False

        if  episode_timesteps + 1  == MAX_EPI_STEPS or done: #To do: Set stop criterion

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
                raw_action = (self.model_offline.select_action(np.array(state),add_noise=False)+ np.random.normal(0, self.max_action * args.expl_noise, size=self.action_dim)).clip(0,self.max_action)
                action = env.action_converter(raw_action)

                

            # Perform action
            # new_action = env.action_converter(action)
            runtime, next_state, reward, done, _ = env.step(action) 
            done_bool = float(done) if episode_timesteps < MAX_EPI_STEPS else 0

            # Store data in replay buffer
            # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            self.replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward
            episode_runtime_list.append(runtime)

        # Train agent after collecting sufficient data
        
            if t >= args.start_timesteps:
                self.model_offline.train(replay_buffer, args.batch_size)  # type: ignore
        else:

            if t < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = self.model_offline.select_action(np.array(state))

                # take action
            next_state, reward, done, _ = env.step(action)
            self.model_offline.store_transition(state, action, reward,next_state)  # type: ignore
            state = next_state
            episode_reward += reward
            episode_reward_list.append(reward)
            
            if t >= args.start_timesteps:
                if self.model_offline.memory_counter > 20:  # type: ignore
                    self.model_offline.train()  # type: ignore
                    
        return self.model_offline


if __name__ == "__main__":
    # Initialize the O2 System with a 5-second interval and a threshold of 0.7
    system = O2System(update_interval=5, threshold=0.7)
    # system.run()

    parser = argparse.ArgumentParser()
    parser.add_argument("--RL_policy", default="DDPG") # Policy name (TD3, DDPG, SAC or DDPG_Context)
    parser.add_argument("--data_file", default='data_11')
    parser.add_argument("--Index", default='ALEX')
    parser.add_argument("--search_method", default='RL', help="method to use")
    parser.add_argument("--search_budget", default=150, type=int)    # search time budget, seconds
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=50, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=100, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1300, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", default=False)              # Save model and optimizer parameters
    parser.add_argument("--load_model", default="default")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--sample_mode",default="random_local")     # random_local: local + global sampling, random: only global sampling
    parser.add_argument("--env_change_freq",default=100)     
    parser.add_argument("--comp_freq",default=200)    
    parser.add_argument("--query_type", default = 'balanced') #Set test query types


    args = parser.parse_args()

    search_time = args.search_budget
    
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
    
    change_i = 0

    
    MAX_EPI_STEPS = 100
    
    if args.Index == "PGM":
        file_name = f"{args.RL_policy}_PGMIndex_{args.seed}"
    elif args.Index == "ALEX":
        file_name = f"{args.RL_policy}_ALEXIndex_{args.seed}"
    elif args.Index == "CARMI":
        file_name = f"{args.RL_policy}_CARMIIndex_{args.seed}"

    
    policy_file = file_name if args.load_model == "default" else args.load_model
    
    online_model = PretrainedRLModel(policy_file=policy_file,args=args)
    offline_model = OfflineRLModel(policy_file=policy_file,args=args)
    
    data_name = "data_11"
    
    evaluations = [eval_policy(online_model, data_file_name, query_type= query_type, search_time=search_time)]


    for t in range(int(args.max_timesteps)):
    
        # Evaluate episode

        if (t+1) % args.comp_freq == 0:
            
            
            data_name_eval = data_name+"_eval"
            
            data_name_train = data_name+"_train"

            print("----------------online evalution---------------")

            [reward_online, _, best_runtime_online, _ ] = eval_policy(online_model, data_name_eval, query_type= query_type, search_time=search_time)

            print("----------------offline evalution---------------")
            [reward_offline, _, best_runtim_offline, _] = eval_policy(offline_model, data_name_eval, query_type= query_type, search_time=search_time)

            if (reward_offline - reward_online)/reward_online >=0.3 and best_runtime_online >=best_runtim_offline:

                print("changed with offline models")

                offline_model.save_model(f"./rlmodels/{file_name}")
                
                online_model.load_model(f"./rlmodels/{policy_file}")
                
            if (t + 1) % args.eval_freq == 0:
                print(f"------Now we are evaluating RL on {data_name_eval}-------")
                evaluations.append(eval_policy(online_model, data_name_eval,query_type= query_type, search_time=search_time))
                if args.Index == "PGM":
                    np.save(f"./results/{args.search_method}/O2_{file_name}_{data_name_eval}", evaluations)
                if args.Index == "ALEX":
                    np.save(f"./results/ALEX/{args.search_method}/O2_{file_name}_{data_name_eval}_{query_type}", evaluations)

                if args.Index == "CARMI":
                    np.save(f"./results/CARMI/{args.search_method}/O2_{file_name}_{data_name_eval}_{query_type}", evaluations)
                    
            if (t + 1) % args.env_change_freq ==0  and change_i <=9:

                change_i += 1
                data_name_train = "data_" + f"{11+change_i}"
                data_file_name = data_name+"_train"

                if args.Index == "PGM":
                    env = PGMIndex(data_file_name)
                    state, done = env.reset(), False
                if args.Index == "ALEX":
                    env = ALEXIndex(data_file_name,query_type=query_type)
                    state, done = env.reset(), False
                if args.Index == "CARMI":
                    env = CARMIIndex(data_file_name,query_type=query_type)
                    state, done = env.reset(), False

                print(f"------Now we are training RL on {data_file_name}-------")
                
            offline_model.refine(data_file_name,args)


        