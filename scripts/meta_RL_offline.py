import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
import copy
from torch import optim
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gym
from agents import TD3, DDPG, dqn, DDPG_Context
from utils import utils
from envs.env import PGMIndex, ALEXIndex, CARMIIndex

def sample_new_environment(base_env, task_index, base_data_path, num_tasks):
    # Generate a new data filename based on the task index
    # Assuming data files are named sequentially as data_0.txt, data_1.txt, ..., data_n.txt
    data_file_name = f"{base_data_path}_data_{task_index % num_tasks}.txt"
    
    # Depending on the type of base environment, initialize a new environment with the new data file
    if isinstance(base_env, PGMIndex):
        new_env = PGMIndex(data_file_name)
    elif isinstance(base_env, ALEXIndex):
        new_env = ALEXIndex(data_file_name, query_type=base_env.query_type)  # Assuming query_type is a property of the environment
    elif isinstance(base_env, CARMIIndex):
        new_env = CARMIIndex(data_file_name, query_type=base_env.query_type)

    return new_env




def eval_policy(policy, data, query_type, eval_episodes=4):
    if args.Index == "ALEX":
        eval_env = ALEXIndex(data, query_type=query_type)
    elif args.Index == "PGM":
        eval_env = PGMIndex(data)
    elif args.Index == "CARMI":
        eval_env = CARMIIndex(data, query_type=query_type)

    eval_env.reset()
    avg_reward = 0.
    best_runtime = np.inf
    for i in range(eval_episodes):
        state, done = eval_env.reset(), False
        eval_env.seed(100 + i)
        for _ in range(5):
            raw_action = policy.select_action(np.array(state))
            action = eval_env.action_converter(raw_action)
            runtime, state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            if runtime < best_runtime:
                best_runtime = runtime
                best_action = action

    avg_reward /= (eval_episodes * 5)
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.5f} parameter: {best_action} best runtime: {best_runtime:.5f}")
    print("---------------------------------------")
    return avg_reward

def adapt_policy(policy, env, num_episodes):
    task_policy = copy.deepcopy(policy)
    adaptation_lr =0.7
    optimizer = optim.Adam(task_policy.parameters(), lr=adaptation_lr)
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = task_policy.select_action(np.array(state))
            next_state, reward, done, _ = env.step(action)
            task_policy.update(state, action, reward, next_state, done)  # Assuming an update function
            state = next_state
        optimizer.zero_grad()
        loss = task_policy.calculate_loss()  # You'll need to define how to calculate this
        loss.backward()
        optimizer.step()
    return task_policy

def main(args):
    # Environment setup
    if args.Index == "PGM":
        env = PGMIndex(args.data_file + ".txt")
    elif args.Index == "ALEX":
        env = ALEXIndex(args.data_file, query_type=args.query_type)
    elif args.Index == "CARMI":
        env = CARMIIndex(args.data_file, query_type=args.query_type)

    # Initialize policy based on the chosen RL algorithm
    if args.RL_policy == "DDPG":
        policy = DDPG.DDPG(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
                       max_action=float(env.action_space.high[0]), discount=args.discount, tau=args.tau)  # Simplified initialization
    else:
        policy = DDPG_Context.DDPG(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
                    max_action=float(env.action_space.high[0]), discount=args.discount, tau=args.tau)  # Simplified initialization
    
    

    # Meta-Learning specific settings
    meta_optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    num_tasks = 10  # Number of different tasks
    num_adaptations = 5  # Adaptation steps

    for task in range(num_tasks):
        # Generate a new task environment from the base environment
        task_env = sample_new_environment(env, task, 'data', num_tasks)
        adapted_policy = adapt_policy(policy, task_env, num_adaptations)

        # Perform evaluations and meta-updates...
        task_loss = eval_policy(adapted_policy, 'data_' + str(task) + '.txt', args.query_type)
        meta_optimizer.zero_grad()
        task_loss.backward()
        meta_optimizer.step()
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
    main(args)
