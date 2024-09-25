import gym
from gym import spaces
import numpy as np
import os
import time
from torch import dtype
import Index.PGM.Parameter_change as pgm_parameter
import Index.Alex.Parameter_change as alex_parameter
import Index.CARMI.Parameter_change as carmi_parameter
import subprocess
from copy import deepcopy
import itertools
import threading
import numpy as np
from typing import Tuple
from typing import Optional
from collections import namedtuple
import psutil
import gym
from gym import spaces
from gym.utils import seeding

TIME_OUT = 300

class SegFaultDetector:
    
    def __init__(self):
        self.main_thread_alive = True  # a flag to keep track of the main thread's status
        self.detector_thread = threading.Thread(target=self._monitor_main_thread)
        self.detector_thread.start()

    def _monitor_main_thread(self):
        """
        Monitor the health of the main thread.
        If the main thread dies, change the flag.
        """
        main_thread = threading.current_thread()
        while self.main_thread_alive:
            if not main_thread.is_alive():
                self.main_thread_alive = False
            time.sleep(0.1)  # polling interval

    def is_main_thread_alive(self):
        """
        Return the status of the main thread.
        """
        return self.main_thread_alive
    

class LinearFitting(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self, train_data):
        super(LinearFitting, self).__init__()
        # Define action and observation space
        # They must be spaces objects
        # Example when using discrete actions:
        self.x = train_data[0]
        self.y = train_data[1]
        self.k_dict = np.arange(0,5,0.01)
        
        self.para_dict = {'k': 1, 'b': 2} # slope and intercept
        self.action_space = spaces.Dict({"k": spaces.Discrete(len(self.k_dict)),"b": spaces.Box(0, 5, shape=(1,))})
        # self.action_space = spaces.Box(low=np.array([0,0],dtype=np.float32),
                            #    high=np.array([5,5],dtype=np.float32),
                            #    )
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=5,
                                            shape=(2,), dtype=np.float32)
        
    def get_k_dict(self):
        return self.k_dict

    def step(self, action):

        self.para_dict["k"] = action[0]
        self.para_dict["b"] = action[1]
        reward = -self.loss()
        done = False

        info = {}

        return np.array([self.para_dict["k"],self.para_dict["b"]]).astype(np.float32) , reward, done, info

    def loss(self):

        total_cost = 0
        M = len(self.x)

        for i in range(M):

            x = self.x[i]
            y = self.y[i]

            total_cost += (y - self.para_dict["k"]* x - self.para_dict["b"]) ** 2

        return total_cost/M

    def reset(self):

        self.para_dict["k"] = 1
        self.para_dict["b"] = 2

        return np.array([self.para_dict["k"],self.para_dict["b"]]).astype(np.float32)  # reward, done, info can't be included

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        pass

    def close (self):
        pass

class PGMIndex(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self, data_file_name):
        super(PGMIndex, self).__init__()
        # Define action and observation space
        # They must be spaces objects
        # Example when using discrete actions:

        self.data_file_name = data_file_name
        
        self.para_dict = {'epsilon': 64, 'er': 4} # PGM tunable parameters

        self.action_dict = {
            'epsilon': list(np.arange(100,1000,100)),
            'er': list(np.arange(0,10,1))
            }

        action_comb =  itertools.product(self.action_dict["epsilon"],self.action_dict["er"])
        self.list_action_comb = [k for k in action_comb]

        action_dim = len(self.list_action_comb)
        state_dim = len(self.list_action_comb)

        self.action_space = spaces.Discrete(action_dim)

        # self.action_space = spaces.Tuple([spaces.Discrete(8000),spaces.Discrete(80)])

        # Example for using image as input (channel-first; channel-last also works):

        # space= {"epsilon": spaces.Box(0,8000,shape=(1,),dtype=int),"er":spaces.Box(5,100,shape=(1,),dtype=int)}

        self.observation_space = spaces.Discrete(state_dim)

        # self.observation_space = spaces.Box(low=1, high=80, shape=(2,), dtype=np.int32)

  

    def step(self, action):

        self.para_dict["epsilon"] = self.list_action_comb[action][0]
        self.para_dict["er"] =  self.list_action_comb[action][1]
        reward = -self.model_loss()
        done = False

        info = {}

        return np.array((self.para_dict["epsilon"],self.para_dict["er"])), reward, done, info

    def model_loss(self):
        
        pgm_parameter.updateFile("./Index/PGM/index_test.cpp",self.para_dict["epsilon"],self.para_dict["er"])
        os.system('g++ ./Index/PGM/index_test.cpp  -w -std=c++17 -o ./Index/PGM/exe_pgm_index')
        os.system(f'./Index/PGM/exe_pgm_index ./data/{self.data_file_name}')

            # other Index in progress

        f = open("runtime_result.txt",encoding="utf-8")
        cost = float(f.read())
        f.close
            
        return cost 

    def reset(self):

        self.para_dict["epsilon"] = 64
        self.para_dict["er"] = 4
        pgm_parameter.updateFile("./Index/PGM/index_test.cpp",self.para_dict["epsilon"],self.para_dict["er"])
        os.system('g++ ./Index/PGM/index_test.cpp  -w -std=c++17 -o ./Index/PGM/exe_pgm_index')
        os.system(f'./Index/PGM/exe_pgm_index ./data/{self.data_file_name}')

        return np.array((self.para_dict["epsilon"],self.para_dict["er"])).astype(np.int32)  # reward, done, info can't be included

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        pass

    def close (self):
        pass
    


class Action:
    """"
    Action class to store and standardize the action for the environment.
    """
    def __init__(self, id_: int, parameters: list):
        """"
        Initialization of an action.

        Args:
            id_: The id of the selected action.
            parameters: The parameters of an action.
        """
        self.id = id_
        self.parameters = parameters

    @property
    def parameter(self):
        """"
        Property method to return the parameter related to the action selected.

        Returns:
            The parameter related to this action_id
        """
        if len(self.parameters) == 2:
            return self.parameters[self.id]
        else:
            return self.parameters[0]


class ALEXIndex(gym.Env):
    """"
    Gym environment parent class.
    ALEXIndex Env
    """
    def __init__(
            self,
            data_file_name,
            query_type
    ):
        """Initialization of the gym environment.

        State: (15 dim)
            no_model_nodes
            no_model_node_expansions
            no_model_node_split
            num_model_node_expansion_pointers
            num_model_node_split_pointers
            no_data_nodes
            no_expand_and_scale
            no_expand_and_retrain
            no_downward_split
            no_sideways_split
            no_downward_split_keys
            no_sideways_split_keys
            no_search
            no_inserts
            no_node_traveral

        Action:

        // User-changeable parameters  (has original setter function)
            external_expectedInsertFrac: continuous [0:1]
            external_maxNodeSize_factor: discrete [10,30] (2^10-2^30)
            external_approximateModelComputation: bool [0,1] (T/F)
            external_approximateCostComputation:  bool [0,1] (T/F)
        //Experimental parameters (may break the system)
            external_fanoutSelectionMethod: discrete [0,1]
            external_splittingPolicyMethod: discrete [0,1]
            external_allowSplittingUpwards: bool [0,1] (T/F)
        //Constant parameters in ALEX
            external_kMinOutOfDomainKeys:discrete [0,N-1]
            external_kinterval: discrete [1000,4000]
            external_kOutOfDomainToleranceFactor: discrete [1,N]
            external_kMinDensity: continuous [0,1]
            external_kDensityinterval_1: continuous [0,0.5]
            external_kDensityinterval_2: continuous [0,0.5]
            external_kAppendMostlyThreshold: continuous [0,1]

        """
        self.data_file_name = data_file_name
        
        self.state_dict = {'no_model_nodes':69, 
                           'no_model_node_expansions':155,
                           'no_model_node_split':0,
                           'num_model_node_expansion_pointers':8388902,
                           'num_model_node_split_pointers':0,
                           'no_data_nodes':3809,
                           'no_expand_and_scale':21000,
                           'no_expand_and_retrain':7196,
                           'no_downward_split':69,
                           'no_sideways_split':3732,
                           'no_downward_split_keys':40977,
                           'no_sideways_split_keys':1439133,
                           'no_search':40000,
                           'no_inserts':1000000,
                           'no_node_traveral':-713000
                            } # AlEX States dict

        self.external_maxNodeSize_factor_list = np.arange(20,30)
        self.external_kMinOutOfDomainKeys_list = np.arange(0,500,5)
        self.kinterval_list = np.arange(1000,5000,100)
        self.external_kOutOfDomainToleranceFactor_list = np.arange(1,51)

        self.action_space = spaces.Dict({
                                        "external_expectedInsertFrac": spaces.Box(0, 1, shape=(1,)),
                                        "external_maxNodeSize_factor": spaces.Discrete(10),
                                        "external_approximateModelComputation": spaces.Discrete(2),
                                        "external_approximateCostComputation":spaces.Discrete(2),
                                        "external_fanoutSelectionMethod":spaces.Discrete(1),
                                        "external_splittingPolicyMethod":spaces.Discrete(1),
                                        "external_allowSplittingUpwards":spaces.Discrete(1),
                                        "external_kMinOutOfDomainKeys":spaces.Discrete(100),
                                        "external_kinterval":spaces.Discrete(40),
                                        "external_kOutOfDomainToleranceFactor":spaces.Discrete(50),
                                        "external_kMinDensity":spaces.Box(0, 1, shape=(1,),dtype=np.float64),
                                        "external_kDensityinterval_1":spaces.Box(0, 1, shape=(1,),dtype=np.float64),
                                        "external_kDensityinterval_2":spaces.Box(0, 1, shape=(1,),dtype=np.float64),
                                        "external_kAppendMostlyThreshold":spaces.Box(0,1,shape=(1,),dtype=np.float64)
                                        })#AlEX Tunable parameters

        self.observation_space = spaces.Box(np.ones(15), -np.ones(15))
        self.query_type = query_type
        self.init_action = np.array([1,24,1,1,0,0,0,5,995,2,0.7,0,0.1,0.9])
        self.init_cost = self.model_cost(self.init_action)
        
        self.last_cost = self.init_cost
        self.imp_count = 0
        
        # Uncomment lines below during training for risk mitigation purposes
        # Memory Leak Detector
        # self.previous_memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB
        
        # # Segmentation Fault Detector
        # self.seg_fault_detector = SegFaultDetector()
        
        # # For runtime check
        # self.allowed_runtime = 60  # set some reasonable limit, e.g., 60 seconds
        # self.start_time = None  # will be set at the start of each step
        

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)  # noqa
        return [seed]

    def action_discretization(self,raw_action_i,n):

        return int(raw_action_i * n) - 1
    
    
    def is_system_in_danger(self):
        """
        Check system-level symptoms indicating issues like endless loops, long runtimes, memory leaks, or segmentation faults.
        """
        # Check for long runtimes
        current_time = time.time()
        if current_time - self.start_time > self.allowed_runtime:
            return True
        
        # Check for memory leak
        current_memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB
        memory_diff = current_memory_usage - self.previous_memory_usage
        if memory_diff > 100:  # assuming 100 MB abrupt increase is an indicator
            return True

        # Check for segmentation fault
        if not self.seg_fault_detector.is_main_thread_alive():
            return True
        
        return False

    def is_forbidden(self, state):
        """
        Check if the state violates some predefined constraints.
        """
        
        # Uncomment lines below during training for risk mitigation purposes
        # # Hard thresholds for ALEX (env1)
        # alex_hard_thresholds = {
        #     "no_model_nodes": 30,  # Somewhat above the provided dangerous value
        #     "no_model_node_expansions": 1500000,
        #     "num_model_node_expansion_pointers": 2400000,
        #     "no_data_nodes": 125,
        #     "no_expand_and_scale": 22000000,
        #     "no_downward_split": 4100000,
        #     "no_sideways_split": 1900000,
        #     "no_downward_split_keys": 17000000,
        #     "no_sideways_split_keys": 115000000,
        #     "no_search": 120000000,
        #     "no_inserts": 31000000,
        #     "no_node_traveral": 140000000
        # }

        # # Linear combinations of specific state variables
        # # If the weighted sum of some states goes beyond a threshold, it indicates danger
        # weighted_sums = [
        #     (state["no_model_nodes"] + 0.5 * state["no_data_nodes"], 45),  # total should not exceed 45
        #     (0.2 * state["no_downward_split"] + 0.3 * state["no_sideways_split"], 1250000),  # total should not exceed 1,250,000
        #     (0.1 * state["no_search"] + 0.05 * state["no_inserts"], 14000000)  # total should not exceed 14,000,000
        # ]
        
        # # Check hard thresholds
        # for key, value in state.items():
        #     if key in alex_hard_thresholds:
        #         if value >= alex_hard_thresholds[key]:
        #             return True  # Current state is in a dangerous region
        
        # # Check linear combinations
        # for sum_value, threshold in weighted_sums:
        #     if sum_value >= threshold:
        #         return True  # Current state is in a dangerous region

        return False
    
    def action_converter(self,action):

        """
        1. Clip the value, discretization
        2. Mapping to the list
        
        """

        #assume input action is a dict. By sample yes, by NN output no

        a_list = action
        #alist = [k for k in action.values()]
        new_action = np.zeros(len(a_list))
        new_action[0] = action[0]

        a_1 = self.action_discretization(action[1],10)

        new_action[1] = self.external_maxNodeSize_factor_list[a_1]

        if action[2] <=0.5:
            new_action[2] = 0
        else:
            new_action[2] = 1

        if action[3] <=0.5:
            new_action[3] = 0
        else:
            new_action[3] = 1

        # if action[4] <=0.5:
        #     new_action[4] = 0
        # else:
        #     new_action[4] = 1

        new_action[4] = 0

        # if action[5] <= 0.5:
        #     new_action[5] = 0
        # else:
        #     new_action[5] = 1

        new_action[5] = 0

        # if action[6] <=0.5:
        #     new_action[6] = 0
        # else:
        #     new_action[6] = 1

        new_action[6] = 0


        a_7 = self.action_discretization(action[7],100)

        new_action[7] = self.external_kMinOutOfDomainKeys_list[a_7]

        a_8 = self.action_discretization(action[8],40)

        new_action[8] = self.kinterval_list[a_8]

        a_9 = self.action_discretization(action[9],50)

        new_action[9] = self.external_kOutOfDomainToleranceFactor_list[a_9]

        new_action[10] = action[10]

        new_action[11] = action[11]/2

        new_action[12] = action[12]/2

        new_action[13] = action[13]

        return new_action

    def action_validation(self,action):

        assert action[0] >= 0 and action[0] <= 1

        assert action[1] >= 20 and action[1] <= 30 

        assert action[2] == 1 or action[2] == 0

        assert action[3] == 1 or action[3] == 0

        # assert action[4] == 1 or action[4] == 0

        # assert action[5] == 0 or action[5] == 1 

        # assert action[6] == 0 or action[6] == 1

        assert action[4] ==0 and action[5] == 0 and action[6]==0

        assert action[7] >= 0 and action[7] <= 500 

        assert action[8] >= 1000 and action[8] <= 5000 

        assert action[9] >= 1 and action[9] <= 51

        assert action[10] >= 0 and action[10] <= 1
        
        assert action[11] >= 0 and action[11] <= 1

        assert action[12] >= 0 and action[12] <= 1

        assert action[13] >= 0 and action[13] <= 1



    def get_state(self,state_file_path):
        file = open(state_file_path,'r')
        state_list = []
        for line in file.readlines():
            O = line.split(":")
            state_list.append(int(O[1]))

        file.close()   
        state = np.asarray(state_list)
        return state


    def reset(self):

        self.last_cost = self.init_cost

        action_reset = np.array([1,24,1,1,0,0,0,5,995,2,0.7,0,0.1,0.9])

        alex_parameter.updateFile("./Index/Alex/src/parameters.hpp",action_reset)
        # os.system('sh ./Index/Alex/write_to_parameter.sh')
        os.system('g++ ./Index/Alex/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/Alex/exe_alex_index')
        os.system(f'./Index/Alex/exe_alex_index ./data_SOSD/{self.data_file_name} {self.query_type}')

        self.init_cost = self.model_cost(self.init_action)
        self.last_cost = self.init_cost
        
        self.imp_count = 0
    

        return self.get_state('./state_result.txt')

    def step(self, action):
        
        self.start_time = time.time()

        # action = self.action_converter(raw_action)
        self.action_validation(action)
        # alex_parameter.updateFile("./Index/Alex/src/parameters.hpp",action)
        # # os.system('sh ./Index/Alex/write_to_parameter.sh')
        # os.system('g++ ./Index/Alex/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/Alex/exe_alex_index')
        # os.system(f'./Index/Alex/exe_alex_index ./data_SOSD/{self.data_file_name} {self.query_type}')

        #To do: update the reward function here
        current_cost = self.model_cost(action)

        runtime = current_cost

        if current_cost == self.last_cost: #Seg fault here, to do

            perf_gain = self.last_cost - current_cost
            deltac0 = perf_gain / self.init_cost

            reward = -1
            
        else:

            perf_gain = self.last_cost - current_cost
                # deltac0 = perf_gain / self.init_cost
            deltac0 = perf_gain /self.last_cost 

            deltac_init = (self.init_cost-current_cost)/ self.init_cost

            if deltac_init > 0 :
                reward = ((1+deltac_init)**2 - 1) * abs(1 + deltac0)
            else:
                reward = -1 * ((1-deltac_init)**2 - 1) * abs(1 - deltac0)

                
            # reward = deltac0

        # Uncomment lines below during training for risk mitigation purposes
        # danger_penalty = -100

        # # Check for system-level dangers
        # if self.is_system_in_danger():
        #     reward += danger_penalty

        # # Check for forbidden state and terminate episode if true
        # if self.is_forbidden(state):
        #     done = True
        #     reward = danger_penalty
 
        
        if (self.init_cost-current_cost)/ self.init_cost >= 0.75:
            done = True
        else:
            done = False

        self.last_cost = current_cost


        info = {}

        state = self.get_state('./state_result.txt')

        return runtime, state, reward, done, info


    def model_cost(self,action):
        
        alex_parameter.updateFile("./Index/Alex/src/parameters.hpp",action)
        # os.system('sh ./Index/Alex/write_to_parameter.sh')
        os.system('g++ ./Index/Alex/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/Alex/exe_alex_index')
        # os.system(f'./Index/Alex/exe_alex_index ./data_SOSD/{self.data_file_name} {self.query_type}')
        cmd = f'./Index/Alex/exe_alex_index ./data_SOSD/{self.data_file_name} {self.query_type}'

        # other Index in progress
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
                    
            return 300

        f = open("./runtime_result.txt",encoding="utf-8")
        cost = float(f.read())
        f.close
            
        return cost 
    


class CARMIIndex(gym.Env):
        """"
        Gym environment parent class.
        CARMIIndex Env
        """
        def __init__(
            self,
            data_file_name,
            query_type
        ):
            self.data_file_name = data_file_name
            
            self.state_dict = {'no_leaf':4951,
                               'lambda':48.6653,
                               'prefetchEnd':-1,
                               'querySize':202000,
                               'researvedSpace':66536,
                               'isInitMode':0,
                               'noInsertQueryKeyVisit':101000,
                               'remainingNode':4804,
                               'scanLeafRange':4804,
                               'noFindQueryKeyVisit':101000,
                               'avgfindQueryVisitPerKey':1,
                               'costSize':219,
                               'avgTimeCost':0.0402169,
                               'avgSpaceCost':0.000798195,
                               'avgTotalCost':0.0790613
                                } # CARMI States dict


            self.external_kMaxLeafNodeSize_factor_list = np.arange(5,15)
            self.external_kMaxLeafNodeSizeExternal_factor_list = np.arange(5,15)

            kMaxLeafNodeSize = spaces.Discrete(10)  
            kMaxLeafNodeSizeExternal = spaces.Discrete(10)  

            # kAlgorithmThreshold = spaces.Box(low=1, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32)

            kAlgorithmThreshold_gap = spaces.Box(low=1, high=3e5, shape=(1,), dtype=np.int32)


            kMemoryAccessTime = spaces.Box(low=1, high=5e2, shape=(1,), dtype=np.float64)
            kLRRootTime = spaces.Box(low=1, high=5e2, shape=(1,), dtype=np.float64)
            kPLRRootTime = spaces.Box(low=1, high=5e2, shape=(1,), dtype=np.float64)
            kLRInnerTime_gap = spaces.Box(low=1, high=5e2, shape=(1,), dtype=np.float64)
            kPLRInnerTime_gap = spaces.Box(low=1, high=5e2, shape=(1,), dtype=np.float64)
            kHisInnerTime_gap = spaces.Box(low=1, high=5e2, shape=(1,), dtype=np.float64)
            kBSInnerTime_gap = spaces.Box(low=1, high=5e2, shape=(1,), dtype=np.float64)
            kCostMoveTime = spaces.Box(low=1, high=5e2, shape=(1,), dtype=np.float64)
            kLeafBaseTime_gap = spaces.Box(low=1, high=5e2, shape=(1,), dtype=np.float64)
            kCostBSTime = spaces.Box(low=1, high=5e2, shape=(1,), dtype=np.float64)


            # kMemoryAccessTime = spaces.Box(low=0, high=np.finfo(np.float64).max, shape=(1,), dtype=np.float64)
            # kLRRootTime = spaces.Box(low=0, high=np.finfo(np.float64).max, shape=(1,), dtype=np.float64)
            # kPLRRootTime = spaces.Box(low=0, high=np.finfo(np.float64).max, shape=(1,), dtype=np.float64)
            # kLRInnerTime = spaces.Box(low=0, high=np.finfo(np.float64).max, shape=(1,), dtype=np.float64)
            # kPLRInnerTime = spaces.Box(low=0, high=np.finfo(np.float64).max, shape=(1,), dtype=np.float64)
            # kHisInnerTime = spaces.Box(low=0, high=np.finfo(np.float64).max, shape=(1,), dtype=np.float64)
            # kBSInnerTime = spaces.Box(low=0, high=np.finfo(np.float64).max, shape=(1,), dtype=np.float64)
            # kCostMoveTime = spaces.Box(low=0, high=np.finfo(np.float64).max, shape=(1,), dtype=np.float64)
            # kLeafBaseTime = spaces.Box(low=0, high=np.finfo(np.float64).max, shape=(1,), dtype=np.float64)
            # kCostBSTime = spaces.Box(low=0, high=np.finfo(np.float64).max, shape=(1,), dtype=np.float64)

            # lambda_low = np.array([-1.0], dtype=np.float64)
            # lambda_high = np.array([100.0], dtype=np.float64)
            # external_lambda = spaces.Box(low=lambda_low, high=lambda_high, dtype=np.float64)
            external_lambda_int = spaces.Discrete(2)  # Two possible values: 0 (representing -1) and 1 (representing a float)
            external_lambda_float = spaces.Box(low=0.1, high=100.0, shape=(1,), dtype=np.float64)

            self.action_space = spaces.Dict({
                'kMaxLeafNodeSize': kMaxLeafNodeSize,
                'kMaxLeafNodeSizeExternal': kMaxLeafNodeSizeExternal,
                'kAlgorithmThreshold': kAlgorithmThreshold_gap,
                'kMemoryAccessTime': kMemoryAccessTime,
                'kLRRootTime': kLRRootTime,
                'kPLRRootTime': kPLRRootTime,
                'kLRInnerTime': kLRInnerTime_gap,
                'kPLRInnerTime': kPLRInnerTime_gap ,
                'kHisInnerTime': kHisInnerTime_gap,
                'kBSInnerTime': kBSInnerTime_gap,
                'kCostMoveTime': kCostMoveTime,
                'kLeafBaseTime': kLeafBaseTime_gap,
                'kCostBSTime': kCostBSTime,
                'external_lambda_int': external_lambda_int,
                'external_lambda_float':external_lambda_float
            })

            self.observation_space = spaces.Box(np.ones(15), -np.ones(15))
            self.init_action = np.array([8,10,58975,80.09,11.54,29.62,5.23,22.8,18.44,26.38,6.25,25.4,10.9438,0,1.0])
            self.query_type = query_type
            self.init_cost = self.model_cost(self.init_action)
            self.last_cost = self.init_cost
            self.imp_count = 0
            
            
            # Uncomment these during training for risk mitigation purposes
            # # Memory Leak Detector
            # self.previous_memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB
            
            # # Segmentation Fault Detector
            # self.seg_fault_detector = SegFaultDetector()
            
            # # For runtime check
            # self.allowed_runtime = 60  # set some reasonable limit, e.g., 60 seconds
            # self.start_time = None  # will be set at the start of each step

    
        def print_f(self):
            print(self.query_type)

        def seed(self, seed: Optional[int] = None):
            self.np_random, seed = seeding.np_random(seed)  # noqa
            return [seed]

        def action_discretization(self,raw_action_i,n):

            return int(raw_action_i * n) - 1
        
        def map_value(self,value, min_range, max_range):
            assert 0.0 <= value <= 1.0, "Value should be in the range (0, 1)"
            return np.float64(min_range) + np.float64(value) * (np.float64(max_range) - np.float64(min_range))
        

        def action_converter(self,action):

            """
            1. Clip the value, discretization
            2. Mapping to the list
            
            """

            #assume input action is a dict. By sample yes, by NN output no

            a_list = action
            #alist = [k for k in action.values()]
            new_action = np.zeros(len(a_list))
            a_0 = self.action_discretization(action[0],10)
            
            new_action[0] = self.external_kMaxLeafNodeSize_factor_list[a_0]
            a_1 = self.action_discretization(action[1],10)

            new_action[1] = self.external_kMaxLeafNodeSizeExternal_factor_list[a_1]

            # a_2_low_bound = max(new_action[0],new_action[1])+1

            new_action[2] = int(self.map_value(action[2],1,3e5))
            
            new_action[3] = self.map_value(action[3],1,5e2)

            new_action[4] = self.map_value(action[4],1,5e2)

            new_action[5] = self.map_value(action[5],1,5e2)
            
            new_action[6] = self.map_value(action[6],1,5e2)

            new_action[7] = self.map_value(action[7],1,5e2)

            new_action[8] = self.map_value(action[8],1,5e2)

            new_action[9] = self.map_value(action[9],1,5e2)
            
            new_action[10] = self.map_value(action[10],1,5e2)

            new_action[11] = self.map_value(action[11],1,5e2)
            
            new_action[12] = self.map_value(action[12],1,5e2)
            

            if action[13] <= 0.5:
                new_action[13] = 0
            else:
                new_action[13] = 1

            new_action[14] = self.map_value(action[14],0.1,100)

            return new_action

        def action_validation(self,action):

            assert action[0] >= 5 and action[0] <= 15

            assert action[1] >= 5 and action[1] <= 15

            assert action[2] >= 1 and action[2] <= 300000

            for i in range(3,13):

                assert action[i]>= 1 and action[i]<=500

            assert action[13]  == 0 or action[13] == 1

            assert action[14]  >= 0.1  and action[14] <= 100
            

        def is_system_in_danger(self):
            """
            Check system-level symptoms indicating issues like endless loops, long runtimes, memory leaks, or segmentation faults.
            """
            # Check for long runtimes
            current_time = time.time()
            if current_time - self.start_time > self.allowed_runtime:
                return True
            
            # Check for memory leak
            current_memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB
            memory_diff = current_memory_usage - self.previous_memory_usage
            if memory_diff > 100:  # assuming 100 MB abrupt increase is an indicator
                return True

            # Check for segmentation fault
            if not self.seg_fault_detector.is_main_thread_alive():
                return True
            
            return False
        
        
        def is_forbidden(self, state):
            """
            Check if the state violates some predefined constraints.
            """
            
            # Uncomment lines below during training for risk mitigation purposes

            # # Hard thresholds for `carmi` environment
            # carmi_hard_thresholds = {
            #     "no_leaf": 1200, 
            #     "lambda": 5600, 
            #     "prefetchEnd": 0,  # Dangerous if it's non-negative
            #     "querySize": 105000,
            #     "reservedSpace": 660000000,
            #     "isInitMode": 2100,
            #     "noFindQueryKeyVisit": 5200,
            #     "avgfindQueryVisitPerKey": 1100,
            #     "noInsertQueryKeyVisit": 5200,
            #     "costSize": 220,
            #     "avgTimeCost": 122000,
            #     "avgSpaceCost": 0.125,
            #     "avgTotalCost": 11,
            #     "remainingNode": 235,
            #     "scanLeafRange": 240
            # }

            # # Linear combinations of specific state variables
            # # If the weighted sum of some states goes beyond a threshold, it indicates danger
            # weighted_sums = [
            #     (state["lambda"] + 0.01 * state["avgTotalCost"], 5611),  # total should not exceed 5611
            #     (0.2 * state["noFindQueryKeyVisit"] + 0.3 * state["noInsertQueryKeyVisit"], 1030),  # total should not exceed 1030
            #     (0.1 * state["avgTimeCost"] + 100 * state["avgSpaceCost"], 12210)  # total should not exceed 12,210
            # ]
            
            # # Check hard thresholds
            # for key, value in state.items():
            #     if key in carmi_hard_thresholds:
            #         if value >= carmi_hard_thresholds[key]:
            #             return True  # Current state is in a dangerous region
            
            # # Check linear combinations
            # for sum_value, threshold in weighted_sums:
            #     if sum_value >= threshold:
            #         return True  # Current state is in a dangerous region

            return False




        def get_state(self,state_file_path):
            file = open(state_file_path,'r')
            state_list = []
            for line in file.readlines():
                O = line.split(":")
                state_list.append(np.float64(O[1]))

            file.close()   
            state = np.asarray(state_list)
            return state


        def reset(self):

            self.last_cost = self.init_cost


            action_reset = np.array([8,10,58975,80.09,11.54,29.62,5.23,22.8,18.44,26.38,6.25,25.4,10.9438,0,1.0])

            carmi_parameter.updateFile("./Index/CARMI/src/parameters.hpp",action_reset)

            os.system('g++ ./Index/CARMI/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/CARMI/exe_carmi_index')
            os.system(f'./Index/CARMI/exe_carmi_index ./data_SOSD/{self.data_file_name} {self.query_type}')

            self.init_cost = self.model_cost(self.init_action)
            self.last_cost = self.init_cost
            
            self.imp_count = 0
            
            return self.get_state('./state_result.txt')

        def step(self, action):
            
            self.start_time = time.time()
            # action = self.action_converter(raw_action)
            self.action_validation(action)
            # carmi_parameter.updateFile("./Index/CARMI/src/parameters.hpp",action)
            # os.system('g++ ./Index/CARMI/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/CARMI/exe_carmi_index')
            # os.system(f'./Index/CARMi/exe_carmi_index ./data_SOSD/{self.data_file_name} {self.query_type}')

            #To do: update the reward function here
            current_cost = self.model_cost(action)

            runtime = current_cost

            if current_cost == self.last_cost: #Seg fault here, to do

                perf_gain = self.last_cost - current_cost
                deltac0 = perf_gain / self.init_cost

                reward = -1
                
            else:

                perf_gain = self.last_cost - current_cost
                # deltac0 = perf_gain / self.init_cost
                deltac0 = perf_gain /self.last_cost 

                deltac_init = (self.init_cost-current_cost)/ self.init_cost

                if deltac_init > 0 :

                    reward = ((1+deltac_init)**2 - 1) * abs(1 + deltac0)
                
                else:

                    reward = -1 * ((1-deltac_init)**2 - 1) * abs(1 - deltac0)
                    
            
            # Uncomment the following code during training for risk mitigation purpose
            # danger_penalty = -100

            # # Check for system-level dangers
            # if self.is_system_in_danger():
            #     reward += danger_penalty

            # # Check for forbidden state and terminate episode if true
            # if self.is_forbidden(state):
            #     done = True
            #     reward = danger_penalty


                # reward = deltac0
            
            if (self.init_cost-current_cost)/ self.init_cost >= 0.75:
                done = True
            else:
                done = False

            self.last_cost = current_cost


            info = {}

            state = self.get_state('./state_result.txt')

            return runtime, state, reward, done, info


        def model_cost(self,action):

            
            carmi_parameter.updateFile("./Index/CARMI/src/parameters.hpp",action)
            os.system('g++ ./Index/CARMI/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/CARMI/exe_carmi_index')
            # os.system(f'./Index/CARMI/exe_carmi_index ./data_SOSD/{self.data_file_name} {self.query_type}')
            cmd = f'./Index/CARMI/exe_carmi_index ./data_SOSD/{self.data_file_name} {self.query_type}'

                # other Index in progress

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
                    
                return 300 

            f = open("./runtime_result.txt",encoding="utf-8")
            cost = float(f.read())
            f.close
                
            return cost 