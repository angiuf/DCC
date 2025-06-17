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
import threading
import json
from pathlib import Path

mp.set_start_method('spawn', force=True)
torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
torch.set_num_threads(1)

# Base paths
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent.parent

def count_collisions(solution, obstacle_map):
    """Count agent-agent and obstacle collisions in the solution."""
    agent_agent_collisions = 0
    obstacle_collisions = 0
    num_agents = 0
    
    if len(solution) > 0:
        num_agents = len(solution)
        
        # Convert solution format to timestep-based format
        timestep_based_solution = []
        if solution and len(solution[0]) > 0:
            # Find max timestep
            max_timestep = 0
            for agent_path in solution:
                for pos in agent_path:
                    max_timestep = max(max_timestep, pos[2])
                    
            # Initialize timestep-based solution
            timestep_based_solution = [[] for _ in range(max_timestep + 1)]
            
            # Fill in the positions for each timestep
            for agent_idx, agent_path in enumerate(solution):
                positions = {}  # Dictionary to store position at each timestep
                for pos in agent_path:
                    positions[pos[2]] = (pos[0], pos[1])
                
                # Ensure every timestep has a position
                for t in range(max_timestep + 1):
                    if t in positions:
                        timestep_based_solution[t].append(positions[t])
                    elif t > 0 and t-1 in positions:
                        # If missing, use previous position
                        timestep_based_solution[t].append(positions[t-1])
                    else:
                        # Should not happen in proper solutions
                        timestep_based_solution[t].append((-1, -1))
        
        # Now count collisions
        for timestep in range(len(timestep_based_solution)):
            positions_at_timestep = timestep_based_solution[timestep]
            current_agent_positions = []
            
            for agent_idx in range(len(positions_at_timestep)):
                agent_pos = positions_at_timestep[agent_idx]
                
                # Check for obstacle collisions
                if agent_pos[0] >= 0 and agent_pos[1] >= 0:  # Valid position
                    if agent_pos[0] < obstacle_map.shape[0] and agent_pos[1] < obstacle_map.shape[1]:
                        if obstacle_map[agent_pos[0], agent_pos[1]] == 1:
                            obstacle_collisions += 1
                
                # Prepare for agent-agent collision check
                current_agent_positions.append(agent_pos)
            
            # Agent-agent collision check
            for i in range(len(current_agent_positions)):
                for j in range(i+1, len(current_agent_positions)):
                    if current_agent_positions[i] == current_agent_positions[j]:
                        # Count collision for both agents involved
                        agent_agent_collisions += 2
    
    return agent_agent_collisions, obstacle_collisions

def get_csv_logger(model_dir, default_model_name):
    csv_path = Path(model_dir) / f"log-{default_model_name}.csv"
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

def create_folders_if_necessary(path):
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

def create_test(test_env_settings: Tuple = config.test_env_settings, num_test_cases: int = config.num_test_cases):
    '''
    create test set
    '''

    for map_length, num_agents, density in test_env_settings:
        test_set_dir = BASE_DIR / 'test_set'
        test_set_dir.mkdir(exist_ok=True)
        name = test_set_dir / f'{map_length}length_{num_agents}agents_{density}density.pth'
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

    # pool = mp.Pool(torch.cuda.device_count())
    pool = mp.Pool(mp.cpu_count()//2)    

    model_path = BASE_DIR / "results"
    model_path.mkdir(exist_ok=True)
    
    dataset_path = PROJECT_DIR / 'baselines/Dataset'
    model_name = "DCC"

    # Map configurations for testing
    map_configurations = [
        {
            "map_name": "15_15_simple_warehouse",
            "size": 15,
            "n_tests": 200,
            "list_num_agents": [4,8,12,16,20,22]
        },
        {
            "map_name": "50_55_simple_warehouse",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4, 8, 16, 32]
        },
        {
            "map_name": "50_55_long_shelves",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4, 8]
        },
        {
            "map_name": "50_55_open_space_warehouse_bottom",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4, 8, 16]
        }
    ]

    header = ["n_agents", 
              "success_rate", "time", "time_std", "time_min", "time_max",
              "episode_length", "episode_length_std", "episode_length_min", "episode_length_max",
              "total_step", "total_step_std", "total_step_min", "total_step_max",
              "avg_step", "avg_step_std", "avg_step_min", "avg_step_max",
              "max_step", "max_step_std", "max_step_min", "max_step_max",
              "min_step", "min_step_std", "min_step_min", "min_step_max",
              "total_costs", "total_costs_std", "total_costs_min", "total_costs_max",
              "avg_costs", "avg_costs_std", "avg_costs_min", "avg_costs_max",
              "max_costs", "max_costs_std", "max_costs_min", "max_costs_max",
              "min_costs", "min_costs_std", "min_costs_min", "min_costs_max",
              "agent_collision_rate", "agent_collision_rate_std", "agent_collision_rate_min", "agent_collision_rate_max",
              "obstacle_collision_rate", "obstacle_collision_rate_std", "obstacle_collision_rate_min", "obstacle_collision_rate_max",
              "total_collision_rate", "total_collision_rate_std", "total_collision_rate_min", "total_collision_rate_max"]

    if isinstance(model_range, int):
        state_dict = torch.load(BASE_DIR / config.save_path / f'{model_range}.pth', map_location=DEVICE)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        print(f'----------test model {model_range}----------')

        # Process each map configuration
        for config_item in map_configurations:
            map_name = config_item["map_name"]
            size = config_item["size"]
            n_tests = config_item["n_tests"]
            list_num_agents = config_item["list_num_agents"]
            
            print(f"\nProcessing map: {map_name}")
            
            # Create output directory for results
            output_dir = dataset_path / map_name / "output" / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup CSV logger
            date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            sanitized_map_name = map_name.replace("/", "_").replace("\\", "_")
            csv_filename_base = f'{model_name}_{sanitized_map_name}_{date}'
            csv_file, csv_logger = get_csv_logger(str(model_path), csv_filename_base)
            
            csv_logger.writerow(header)
            csv_file.flush()

            # Process each agent count
            for num_agents in list_num_agents:
                print(f"Starting tests for {num_agents} agents on map {map_name}")
                tests = load_test_set(dataset_path, map_name, num_agents, n_tests, network)

                # Create output directory if it doesn't exist
                output_agent_dir = output_dir / f"{num_agents}_agents"
                output_agent_dir.mkdir(exist_ok=True)

                print("testing...")
                ret = pool.map(test_one_case, tests)

                success, episode_length, time, num_comm, total_steps, avg_step, max_step, min_step, total_costs, avg_costs, max_costs, min_costs, agent_coll_rate, obstacle_coll_rate, total_coll_rate, solution = zip(*ret)

                # save solution
                for i in range(n_tests):
                    out = dict()
                    out["finished"] = success[i]
                    if out["finished"]:
                        out["time"] = time[i]
                        out["episode_length"] = int(episode_length[i])
                        out["total_steps"] = int(total_steps[i])
                        out["avg_steps"] = int(avg_step[i])
                        out["max_steps"] = int(max_step[i])
                        out["min_steps"] = int(min_step[i])
                        out["total_costs"] = int(total_costs[i])
                        out["avg_costs"] = int(avg_costs[i])
                        out["max_costs"] = int(max_costs[i])
                        out["min_costs"] = int(min_costs[i])
                        out["agent_coll_rate"] = agent_coll_rate[i]
                        out["obstacle_coll_rate"] = obstacle_coll_rate[i]
                        out["total_coll_rate"] = total_coll_rate[i]
                        out["crashed"] = (agent_coll_rate[i] + obstacle_coll_rate[i]) > 0
                    out["communication_times"] = int(num_comm[i])

                    solution_filepath = output_agent_dir / f"solution_{model_name}_{map_name}_{num_agents}_agents_ID_{str(i).zfill(3)}.txt"
                    with open(solution_filepath, 'w') as f:
                        f.write("Metrics:\n")
                        json.dump(out, f, indent=4)
                        f.write("\n\nSolution:\n")
                        if solution[i]:
                            for agent_path in solution[i]:
                                f.write(f"{agent_path}\n")
                        else:
                            f.write("No solution found.\n")

                # Calculate aggregated metrics
                final_results = {}
                final_results['finished'] = np.sum(success) / len(success) if len(success) > 0 else 0
                
                # Filter successful cases for metrics
                successful_indices = [i for i, s in enumerate(success) if s]
                
                if successful_indices:
                    successful_episode_length = [episode_length[i] for i in successful_indices]
                    successful_total_steps = [total_steps[i] for i in successful_indices]
                    successful_avg_step = [avg_step[i] for i in successful_indices]
                    successful_max_step = [max_step[i] for i in successful_indices]
                    successful_min_step = [min_step[i] for i in successful_indices]
                    successful_total_costs = [total_costs[i] for i in successful_indices]
                    successful_avg_costs = [avg_costs[i] for i in successful_indices]
                    successful_max_costs = [max_costs[i] for i in successful_indices]
                    successful_min_costs = [min_costs[i] for i in successful_indices]
                    successful_agent_coll_rate = [agent_coll_rate[i] for i in successful_indices]
                    successful_obstacle_coll_rate = [obstacle_coll_rate[i] for i in successful_indices]
                    successful_total_coll_rate = [total_coll_rate[i] for i in successful_indices]
                else:
                    successful_episode_length = []
                    successful_total_steps = []
                    successful_avg_step = []
                    successful_max_step = []
                    successful_min_step = []
                    successful_total_costs = []
                    successful_avg_costs = []
                    successful_max_costs = []
                    successful_min_costs = []
                    successful_agent_coll_rate = []
                    successful_obstacle_coll_rate = []
                    successful_total_coll_rate = []

                print("success rate: {:.2f}%".format(final_results['finished']*100))
                print("episode_length: {}".format(np.mean(successful_episode_length) if successful_episode_length else 0))
                print("total_steps: {}".format(np.mean(successful_total_steps) if successful_total_steps else 0))
                print("avg_step: {}".format(np.mean(successful_avg_step) if successful_avg_step else 0))
                print("max_step: {}".format(np.mean(successful_max_step) if successful_max_step else 0))
                print("communication times: {}".format(np.mean(num_comm) if num_comm else 0))
                print("collision_rate: {}".format(np.mean(successful_total_coll_rate)*100 if successful_total_coll_rate else 0))
                print()

                data = [num_agents, 
                        final_results['finished'] * 100,  # convert to percentage
                        np.mean(time) if time else 0,
                        np.std(time) if time else 0,
                        np.min(time) if time else 0,
                        np.max(time) if time else 0,
                        np.mean(successful_episode_length) if successful_episode_length else 0,
                        np.std(successful_episode_length) if successful_episode_length else 0,
                        np.min(successful_episode_length) if successful_episode_length else 0,
                        np.max(successful_episode_length) if successful_episode_length else 0,
                        np.mean(successful_total_steps) if successful_total_steps else 0,
                        np.std(successful_total_steps) if successful_total_steps else 0,
                        np.min(successful_total_steps) if successful_total_steps else 0,
                        np.max(successful_total_steps) if successful_total_steps else 0,
                        np.mean(successful_avg_step) if successful_avg_step else 0,
                        np.std(successful_avg_step) if successful_avg_step else 0,
                        np.min(successful_avg_step) if successful_avg_step else 0,
                        np.max(successful_avg_step) if successful_avg_step else 0,
                        np.mean(successful_max_step) if successful_max_step else 0,
                        np.std(successful_max_step) if successful_max_step else 0,
                        np.min(successful_max_step) if successful_max_step else 0,
                        np.max(successful_max_step) if successful_max_step else 0,
                        np.mean(successful_min_step) if successful_min_step else 0,
                        np.std(successful_min_step) if successful_min_step else 0,
                        np.min(successful_min_step) if successful_min_step else 0,
                        np.max(successful_min_step) if successful_min_step else 0,
                        np.mean(successful_total_costs) if successful_total_costs else 0,
                        np.std(successful_total_costs) if successful_total_costs else 0,
                        np.min(successful_total_costs) if successful_total_costs else 0,
                        np.max(successful_total_costs) if successful_total_costs else 0,
                        np.mean(successful_avg_costs) if successful_avg_costs else 0,
                        np.std(successful_avg_costs) if successful_avg_costs else 0,
                        np.min(successful_avg_costs) if successful_avg_costs else 0,
                        np.max(successful_avg_costs) if successful_avg_costs else 0,
                        np.mean(successful_max_costs) if successful_max_costs else 0,
                        np.std(successful_max_costs) if successful_max_costs else 0,
                        np.min(successful_max_costs) if successful_max_costs else 0,
                        np.max(successful_max_costs) if successful_max_costs else 0,
                        np.mean(successful_min_costs) if successful_min_costs else 0,
                        np.std(successful_min_costs) if successful_min_costs else 0,
                        np.min(successful_min_costs) if successful_min_costs else 0,
                        np.max(successful_min_costs) if successful_min_costs else 0,
                        np.mean(successful_agent_coll_rate) * 100 if successful_agent_coll_rate else 0,  # convert to percentage
                        np.std(successful_agent_coll_rate) * 100 if successful_agent_coll_rate else 0,
                        np.min(successful_agent_coll_rate) * 100 if successful_agent_coll_rate else 0,
                        np.max(successful_agent_coll_rate) * 100 if successful_agent_coll_rate else 0,
                        np.mean(successful_obstacle_coll_rate) * 100 if successful_obstacle_coll_rate else 0,  # convert to percentage
                        np.std(successful_obstacle_coll_rate) * 100 if successful_obstacle_coll_rate else 0,
                        np.min(successful_obstacle_coll_rate) * 100 if successful_obstacle_coll_rate else 0,
                        np.max(successful_obstacle_coll_rate) * 100 if successful_obstacle_coll_rate else 0,
                        np.mean(successful_total_coll_rate) * 100 if successful_total_coll_rate else 0,  # convert to percentage
                        np.std(successful_total_coll_rate) * 100 if successful_total_coll_rate else 0,
                        np.min(successful_total_coll_rate) * 100 if successful_total_coll_rate else 0,
                        np.max(successful_total_coll_rate) * 100 if successful_total_coll_rate else 0
                       ]
                csv_logger.writerow(data)
                csv_file.flush()
            
            csv_file.close()

class TestCustomEnv:
    counter = 0
    counter_lock = threading.Lock()

    @staticmethod
    def print_episode():
        with TestCustomEnv.counter_lock:
            TestCustomEnv.counter += 1
            # print(f"Current episode: {TestCustomEnv.counter}")

def test_one_case(args):
    env_set, network, num_agents, map_data = args
    TestCustomEnv.print_episode()
    
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
    num_coll = 0
    steps = np.zeros(num_agents)
    costs = np.zeros(num_agents)  # Cost per agent

    time = datetime.datetime.now()

    while not done and env.steps < config.max_episode_length:
        
        actions, _, _, _, comm_mask = network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(last_act.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(pos.astype(int)))
        
        (obs, last_act, pos), rewards, done, _ = env.step(actions)

        episode_length += 1

        # Get the number of collisions (There is a collision if the reward is -0.5)
        for i in range(num_agents):
            if rewards[i] == -0.5:
                num_coll += 1

        for i in range(num_agents):
            if actions[i] != 0:
                steps[i] += 1
            # Update costs - each agent pays 1 per timestep until reaching goal
            if not np.array_equal(pos[i], env.goals_pos[i]):
                costs[i] = episode_length + 1
                
        total_step += np.count_nonzero(actions)
        num_comm += np.sum(comm_mask)

        # Update the solution
        for i in range(num_agents):
            solution[i].append(np.append(pos[i], episode_length))

    time_elapsed = datetime.datetime.now() - time
    time_elapsed = time_elapsed.total_seconds()    
    
    avg_step = total_step / num_agents
    max_step = np.max(steps)
    min_step = np.min(steps)
    
    # Calculate costs
    total_costs = np.sum(costs)
    avg_costs = np.mean(costs)
    max_costs = np.max(costs)
    min_costs = np.min(costs)
    
    # Calculate collision rates using the same method as PRIMAL/AB-MAPPER
    if episode_length > 0 and num_agents > 0:
        agent_coll, obs_coll = count_collisions(solution, map_data)
        agent_coll_rate = agent_coll / (episode_length * num_agents)
        obstacle_coll_rate = obs_coll / (episode_length * num_agents)
        total_coll_rate = (agent_coll + obs_coll) / (episode_length * num_agents)
    else:
        agent_coll_rate = 0
        obstacle_coll_rate = 0
        total_coll_rate = 0

    return (np.array_equal(env.agents_pos, env.goals_pos), episode_length, time_elapsed, num_comm, total_step, avg_step, max_step, min_step, 
            total_costs, avg_costs, max_costs, min_costs, agent_coll_rate, obstacle_coll_rate, total_coll_rate, solution)




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
    map_file = dataset_path / map_name / 'input/map' / f'{map_name}.npy'
    map_data = np.load(map_file)
    for i in tqdm(range(num_test_cases)):
        case_filepath = dataset_path / map_name / 'input/start_and_goal' / f'{num_agents}_agents' / f'{map_name}_{num_agents}_agents_ID_{str(i).zfill(3)}.npy'
        pos = np.load(case_filepath, allow_pickle=True)
        start_pos = pos[:,0]
        goal_pos = pos[:,1]
        tests.append([map_data, start_pos, goal_pos])

    tests = [(test, network, num_agents, map_data) for test in tests]
    return tests

if __name__ == '__main__':
    # load trained model and reproduce results in paper
    test_model_custom_env(128000)
    print("Finished all tests!")