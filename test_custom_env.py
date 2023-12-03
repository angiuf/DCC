'''create test set and test model'''
import os
import random
import pickle
from typing import Tuple, Union
import warnings
warnings.simplefilter("ignore", UserWarning)
from tqdm import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp
from environment import Environment
from model import Network
import config

torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
DEVICE = torch.device('cpu')
torch.set_num_threads(1)

def create_test(test_env_settings: Tuple = config.test_env_settings, num_test_cases: int = config.num_test_cases):
    '''
    create test set
    '''

    for map_length, num_agents, density in test_env_settings:

        name = f'./test_set/{map_length}length_{num_agents}agents_{density}density.pth'
        print(f'-----{map_length}length {num_agents}agents {density}density-----')

        tests = []

        env = Environment(fix_density=density, num_agents=num_agents, map_length=map_length)

        for _ in tqdm(range(num_test_cases)):
            tests.append((np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos)))
            env.reset(num_agents=num_agents, map_length=map_length)
        print()

        with open(name, 'wb') as f:
            pickle.dump(tests, f)



def test_model_custom_env(model_range: Union[int, tuple]):
    '''
    test model in 'saved_models' folder
    '''
    network = Network()
    network.eval()
    network.to(DEVICE)

    pool = mp.Pool(mp.cpu_count()//2)

    if isinstance(model_range, int):
        state_dict = torch.load(os.path.join(config.save_path, f'{model_range}.pth'), map_location=DEVICE)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        
        print(f'----------test model {model_range}----------')

        print(f"testing in custom warehouse environment")
        # with open('./test_set/{}length_{}agents_{}density.pth'.format(40, 4, 0.3), 'rb') as f:
        #     tests = pickle.load(f)
        #     print(len(tests))

        # create test set with random environment
        tests = []
        print("creating test set")
        for _ in tqdm(range(config.num_test_cases)):
            tests.append((get_warehouse_obs(), get_agents_and_goal(config.n_agents)[0], get_agents_and_goal(config.n_agents)[1]))

        tests = [(test, network) for test in tests]

        print("testing...")
        ret = pool.map(test_one_case, tests)

        success, steps, num_comm = zip(*ret)


        print("success rate: {:.2f}%".format(sum(success)/len(success)*100))
        print("average step: {}".format(sum(steps)/len(steps)))
        print("communication times: {}".format(sum(num_comm)/len(num_comm)))
        print()

    elif isinstance(model_range, tuple):

        for model_name in range(model_range[0], model_range[1]+1, config.save_interval):
            state_dict = torch.load(os.path.join(config.save_path, f'{model_name}.pth'), map_location=DEVICE)
            network.load_state_dict(state_dict)
            network.eval()
            network.share_memory()


            print(f'----------test model {model_name}----------')

            print(f"testing in custom warehouse environment")
            with open('./test_set/{}length_{}agents_{}density.pth'.format(40, 4, 0.3), 'rb') as f:
                tests = pickle.load(f)

            tests = [(test, network) for test in tests]
            ret = pool.map(test_one_case, tests)


            success, steps, num_comm = zip(*ret)

            print("success rate: {:.2f}%".format(sum(success)/len(success)*100))
            print("average step: {}".format(sum(steps)/len(steps)))
            print("communication times: {}".format(sum(num_comm)/len(num_comm)))
            print()

            print('\n')
            

def test_one_case(args):

    env_set, network = args

    env = Environment()
    env.load(env_set[0], env_set[1], env_set[2])
    obs, last_act, pos = env.observe()
    
    done = False
    network.reset()

    step = 0
    num_comm = 0
    while not done and env.steps < config.max_episode_length:
        actions, _, _, _, comm_mask = network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(last_act.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(pos.astype(int)))
        (obs, last_act, pos), _, done, _ = env.step(actions)
        step += 1
        num_comm += np.sum(comm_mask)

    return np.array_equal(env.agents_pos, env.goals_pos), step, num_comm




def code_test():
    env = Environment()
    network = Network()
    network.eval()
    obs, last_act, pos = env.observe()
    network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(last_act.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(pos.astype(int)))
    


def get_warehouse_obs():
    return np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      ])

def get_agents_and_goal(n_agents):    
    open_list = [[3, 0],
                 [4, 0],
                 [5, 0],
                 [6, 0],
                 [7, 0],
                 [8, 0],
                 [9, 0],
                 [10, 0],
                 [11, 0],
                 [12, 0], 
                 [3, 14],
                 [4, 14],
                 [5, 14],
                 [6, 14],
                 [7, 14],
                 [8, 14],
                 [9, 14],
                 [10, 14],
                 [11, 14],
                 [12, 14],
                 [2, 4],
                 [2, 6],
                 [2, 8],
                 [2, 10],
                 [4, 4],
                 [4, 6],
                 [4, 8],
                 [4, 10],
                 [6, 4],
                 [6, 6],
                 [6, 8],
                 [6, 10],
                 [8, 4],
                 [8, 6],
                 [8, 8],
                 [8, 10],
                 [10, 4],
                 [10, 6],
                 [10, 8],
                 [10, 10],
                 [12, 4],
                 [12, 6],
                 [12, 8],
                 [12, 10]]
    
    # Error if n_agents > len(open_list)
    if n_agents > len(open_list)/2:
        raise ValueError(f"n_agents %d must be less than or equal to the available pairs of start/goal positions %d" % (n_agents, len(open_list)/2))
    
    start_pos = []
    goal_pos = []
    for i in range(n_agents):
        # Randomly choose a starting point
        start = random.choice(open_list)
        open_list.remove(start)
        # Randomly choose a goal point
        goal = random.choice(open_list)
        open_list.remove(goal)
        start_pos.append(start)
        goal_pos.append(goal)
    
    return np.array(start_pos), np.array(goal_pos)


if __name__ == '__main__':

    # load trained model and reproduce results in paper
    test_model_custom_env(128000)