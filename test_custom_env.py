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

    list_num_agents = [4, 8, 12, 16, 20, 22]
    num_test_cases = 200
    dataset_path = '/home/andrea/Thesis/baselines/Dataset/'
    map_name = '15_simple_warehouse'
    model_name = "DCC"

    # create output folder if not exists
    output_dir = dataset_path + map_name + "/output/" + model_name + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        for num_agents in list_num_agents:
            print("loading test set")
            tests = load_test_set(dataset_path, map_name, num_agents, num_test_cases, network)

            # Create output directory if it doesn't exist
            output_agent_dir = output_dir + str(num_agents) + "_agents" + "/"
            if not os.path.exists(output_agent_dir):
                os.makedirs(output_agent_dir)

            print("testing...")
            ret = pool.map(test_one_case, tests)

            success, episode_length, num_comm, total_steps, avg_step, max_step, solution = zip(*ret)

            # save solution
            for i in range(num_test_cases):
                filepath = output_agent_dir + "solution_" + model_name + "_" + map_name + "_" + str(num_agents) + "_agents_ID_" + str(i).zfill(5) + ".npy"
                np.save(filepath, solution[i])


            print("success rate: {:.2f}%".format(sum(success)/len(success)*100))
            print("episode_length: {}".format(sum(episode_length)/len(episode_length)))
            print("total_steps: {}".format(sum(total_steps)/len(total_steps)))
            print("avg_step: {}".format(sum(avg_step)/len(avg_step)))
            print("max_step: {}".format(sum(max_step)/len(max_step)))
            print("communication times: {}".format(sum(num_comm)/len(num_comm)))
            print()

            header = ["n_agents", "success_rate", "total_step", "avg_step", "max_step", "episode_length", "communication_times"]
            data = [num_agents, sum(success)/len(success)*100, sum(total_steps)/len(total_steps), sum(avg_step)/len(avg_step), sum(max_step)/len(max_step), sum(episode_length)/len(episode_length), sum(num_comm)/len(num_comm)]
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

    # Initialize a dict solution with the starting positions
    solution = env.agents_pos
    solution = [[np.append(pos, 0)] for pos in solution]

    episode_length = 0
    total_step = 0
    num_comm = 0
    steps = np.zeros(num_agents)

    while not done and env.steps < config.max_episode_length:
        actions, _, _, _, comm_mask = network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(last_act.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(pos.astype(int)))
        (obs, last_act, pos), _, done, _ = env.step(actions)

        for i in range(num_agents):
            if actions[i] != 0:
                steps[i] += 1
        episode_length += 1
        total_step += np.count_nonzero(actions)
        num_comm += np.sum(comm_mask)

        # Update the solution
        for i in range(num_agents):
            solution[i].append(np.append(pos[i], episode_length))
    
    avg_step = total_step / num_agents
    max_step = np.max(steps)

    return np.array_equal(env.agents_pos, env.goals_pos), episode_length, num_comm, total_step, avg_step, max_step, solution




def code_test():
    env = Environment()
    network = Network()
    network.eval()
    obs, last_act, pos = env.observe()
    network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(last_act.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(pos.astype(int)))
    


def load_test_set(dataset_path, map_name, num_agents, num_test_cases, network):
    tests = []
    map = np.load(dataset_path + map_name + '/input/map/' + map_name + '.npy')
    for i in tqdm(range(num_test_cases)):
        case_filepath = dataset_path + '/' + map_name + '/input/start_and_goal/' + str(num_agents) + '_agents/' + map_name + '_' + str(num_agents) + '_agents_ID_' + str(i).zfill(5) + '.npy'
        start_pos, goal_pos = np.load(case_filepath, allow_pickle=True)
        tests.append([map, start_pos, goal_pos])

    tests = [(test, network, num_agents) for test in tests]
    return tests

if __name__ == '__main__':

    # load trained model and reproduce results in paper
    test_model_custom_env(128000)