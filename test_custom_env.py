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
import datetime
import csv

torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
DEVICE = torch.device('cpu')
torch.set_num_threads(1)

def get_csv_logger(model_dir, default_model_name):
    csv_path = os.path.join(model_dir, "log-"+default_model_name+".csv")
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

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

    model_path = "./final/"
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    model_name = 'evaluation_custom_warehouse_SCRIMP_' + date
    csv_file, csv_logger = get_csv_logger(model_path, model_name)

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
        list_num_agents = [4, 8, 12, 16, 20, 22]
        for num_agents in list_num_agents:
            tests = []
            print("creating test set")
            for _ in tqdm(range(config.num_test_cases)):
                tests.append((get_warehouse_obs(), get_agents_and_goal(num_agents)[0], get_agents_and_goal(num_agents)[1]))

            tests = [(test, network, num_agents) for test in tests]

            print("testing...")
            ret = pool.map(test_one_case, tests)

            success, episode_length, num_comm, total_steps, avg_step = zip(*ret)


            print("success rate: {:.2f}%".format(sum(success)/len(success)*100))
            print("episode_length: {}".format(sum(episode_length)/len(episode_length)))
            print("total_steps: {}".format(sum(total_steps)/len(total_steps)))
            print("avg_step: {}".format(sum(avg_step)/len(avg_step)))
            print("communication times: {}".format(sum(num_comm)/len(num_comm)))
            print()

            header = ["n_agents", "success_rate", "total_step", "avg_step", "episode_length", "communication_times"]
            data = [num_agents, sum(success)/len(success)*100, sum(total_steps)/len(total_steps), sum(avg_step)/len(avg_step), sum(episode_length)/len(episode_length), sum(num_comm)/len(num_comm)]
            if num_agents == 4:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

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

    env_set, network, num_agents = args

    env = Environment()
    env.load(env_set[0], env_set[1], env_set[2])
    obs, last_act, pos = env.observe()
    
    done = False
    network.reset()

    episode_length = 0
    total_step = 0
    num_comm = 0
    while not done and env.steps < config.max_episode_length:
        actions, _, _, _, comm_mask = network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(last_act.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(pos.astype(int)))
        (obs, last_act, pos), _, done, _ = env.step(actions)

        episode_length += 1
        total_step += np.count_nonzero(actions)
        num_comm += np.sum(comm_mask)
    
    avg_step = total_step / num_agents

    return np.array_equal(env.agents_pos, env.goals_pos), episode_length, num_comm, total_step, avg_step




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