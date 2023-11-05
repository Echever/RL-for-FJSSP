from src.env import FJSSPEnv
from src.generator import generate_instance_list
from src.parsedata import get_data, parse
import json
import torch
from datetime import datetime
import random
from src.ppo import PPO
import os
import numpy as np
import time


os.makedirs('candidate_models', exist_ok=True)
os.makedirs('models', exist_ok=True)

def generate_train_instances(train_config):
    list_instances = generate_instance_list(**train_config)
    instances = []
    for instance in list_instances:
        jobs, operations, info, maximum = get_data(parse(instance))
        instances.append({ "jobs": jobs, "operations": operations, "maximum": maximum, "num_machines": info["machinesNb"]})
    return instances

def train(max_episodes = 10, train_freq = 2, new_freq=1, n_cases = 100, mask_option=1, sel_k=1, batch_size = 20, lr=0.001, hidden_channels=128, num_layers = 1, heads = 3
         ,j_max = 11, j_min = 9, m_max = 6, m_min = 5, op_max = 6, max_processing = 25  ):

    with open('val/validation_set.json', 'r') as infile:
        validation_set = json.load(infile)

    val_env = FJSSPEnv(validation_set, mask_option, sel_k)
    s = val_env.reset()
    metadata = s.metadata()

    max_episodes = max_episodes
    train_freq = train_freq
    new_freq = new_freq
    lr_actor = lr     # learning rate for actor network
    lr_critic = lr     # learning rate for critic network
    batch_size = batch_size

    lr_actor = lr     # learning rate for actor network
    lr_critic = lr     # learning rate for critic network

    K_epochs = 3              # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 1                # discount factor

    train_config = {
        "n_cases": n_cases,
        "range_jobs": (j_min, j_max),
        "range_machines": (m_min, m_max),
        "range_op_per_job": (4, op_max),
        "max_processing": max_processing
    }

    instances = generate_train_instances(train_config)

    env = FJSSPEnv(instances, mask_option, sel_k)
    ppo_agent = PPO(lr_actor, lr_critic, gamma, K_epochs, eps_clip, env, metadata,  hidden_channels, num_layers, batch_size, heads)

    episode_number = 1
    sample = 0

    # training loop
    while True:
        state = env.reset()
        current_ep_reward = 0
        if episode_number > max_episodes:
            break
        for t in range(1, 10**10):
            action = ppo_agent.select_action(state, sample, t)
            state, reward, done, _ = env.step(action)
            
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
              
            if done:
                current_ep_reward += reward
                if episode_number%train_freq == 0:
                    loss, loss_cri = ppo_agent.update()

                    if episode_number%(new_freq*train_freq) == 0:
                        instances = generate_train_instances(train_config)
                        env = FJSSPEnv(instances, mask_option, sel_k)
                
                break

        episode_number+=1

    env.close()

    best_difference = 0
    imporve_ind = []
    improve_res = []
    val_env = FJSSPEnv(validation_set, mask_option, sel_k)

    with open('val/validation_results.json', 'r') as infile:
        validation_results = json.load(infile)

    all_val_results = []
    for i in range(len(validation_set)):
        v_state = val_env.reset()
        for q in range(1, 10**10):
            v_action = ppo_agent.select_action(v_state, 2, q)
            v_state, v_reward, v_done, _ = val_env.step(v_action)
            if v_done:
                res = val_env.mk/int(validation_set[i]["score"])-1
                all_val_results.append(res)
                if res<validation_results[i]["best"]:
                    imporve_ind.append(i)
                    improve_res.append(res)
                if validation_results[i]["best"] - res>best_difference:
                    best_difference = validation_results[i]["best"] - res
                break
        
    if len(imporve_ind)>0:
        name = str(int(random.uniform(10**10, 10**15)))
        with open('candidate_models/model_params.json', 'r') as infile:
                model_params = json.load(infile)
            
        model_params.append({"name": name+".pth", "sel_k": sel_k, "mask_option": mask_option, "num_layers": num_layers, "hidden_channels": hidden_channels, "heads": heads, "all_val_results": all_val_results})
        with open('candidate_models/model_params.json', 'w') as outfile:
                json.dump(model_params, outfile)
        ppo_agent.save("candidate_models/"+name+".pth")

        for index, l_i in enumerate(imporve_ind):
            validation_results[l_i]["best"] = improve_res[index]
            validation_results[l_i]["name"] = name
        with open('val/validation_results.json', 'w') as outfile:
            json.dump(validation_results, outfile)

    return best_difference

def test_model(model_name, folder, filename):

    folder_path = folder
    test_instances = []
    
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r') as file:
        contents = file.read()
        jobs, operations, info, maximum = get_data(parse(contents))
        test_instances.append({"jobs": jobs, "operations": operations, "maximum": maximum, "num_machines": info["machinesNb"]})

    test_env = FJSSPEnv(test_instances)

    s = test_env.reset()
    metadata = s.metadata()

    models = [model_name] 

    best_results = [10**10]*len(test_instances)

    all_results = []
    with open('models/model_params.json', 'r') as infile:
        model_params = json.load(infile)

    with torch.no_grad():
        for m in models:
            model_results = []
            param = [p for p in model_params if p["name"]==m][0]
            test_env = FJSSPEnv(test_instances, param["mask_option"], param["sel_k"]) ##########
            t_ppo_agent = PPO(0.001, 0.001, 1, 3, 3, test_env, metadata,  param["hidden_channels"], param["num_layers"], 64, param["heads"])
            # preTrained weights directory
            t_ppo_agent.load("models/"+m)

            start_time = time.time()

            for i in range(len(test_instances)):
                v_state = test_env.reset()
                for q in range(1, 10**10):
                    v_action = t_ppo_agent.select_action(v_state, 2, q)
                    v_state, v_reward, v_done, _ = test_env.step(v_action)
                    if v_done:
                        model_results.append(test_env.mk)
                        if test_env.mk< best_results[i]:
                            best_results[i] = test_env.mk
                        break


            all_results.append(model_results)
    

    return best_results, start_time, time.time()
