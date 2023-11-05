import copy
import gym
import numpy as np
import random
from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T

class FJSSPEnv(gym.Env):
    def get_prev_op(self, o_id):
        for j in self.jobs:
            if o_id in j:
                if j.index(o_id) == 0:
                    return None
                else:
                    return j[j.index(o_id) -1]

    def __init__(self, instances, mask_option = 3, sel_k = 5):
        super(FJSSPEnv, self).__init__()
        self.instances = instances
        self.current_instance = 0
        self.mask_option = mask_option
        self.sel_k = sel_k

    def generate_instance(self, instance):
        jobs, operations = instance["jobs"], instance["operations"]

        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.operations = operations
        self.num_operations = len(operations)
        self.num_machines = len(instance["operations"][0])

        self.num_features_job = 4
        self.num_features_oper = 2
        self.num_features_mach = 3

        self.max_change = int(len(operations))
        self.data = HeteroData()

        self.data["job"].x = torch.zeros((self.num_jobs, self.num_features_job), dtype= torch.float)
        self.data["operation"].x = torch.zeros((len(self.operations), self.num_features_oper), dtype= torch.float)
        self.data["machine"].x = torch.zeros((self.num_machines, self.num_features_mach), dtype= torch.float)

        aux_list = []
        for i in range(self.num_jobs):
            for j in self.jobs[i]:
                aux_list.append([j, i])

        self.data['operation', 'belongs', 'job'].edge_index = torch.LongTensor(aux_list).T

        aux_list = []
        for j in self.jobs:
            for i in range(len(j)-1):
                aux_list.append([j[i], j[i]])
                aux_list.append([j[i+1], j[i]])
            aux_list.append([j[i+1], j[i+1]])

        self.data['operation', 'prec', 'operation'].edge_index = torch.LongTensor(aux_list).T

        aux_list = []
        aux_list2 = []
        for j in range(self.num_jobs):
            for i in range(self.num_machines):
                aux_list.append([i, j])
                aux_list2.append([j,i])

        aux_list = []
        for i in range(self.num_machines):
            for j in range(self.num_machines):
                aux_list.append([i, j])
        self.data['machine', 'listens', 'machine'].edge_index = torch.LongTensor(aux_list).T

        aux_list = []
        for i in range(self.num_jobs):
            for j in range(self.num_jobs):
                aux_list.append([i, j])
        self.data['job', 'listens', 'job'].edge_index = torch.LongTensor(aux_list).T

        self.all_pendings  = [] 
        for os in jobs:
            aux = []
            #aux_o = np.array(operations[j])
            for o_id in reversed(os):
                aux_o = np.array(operations[o_id])
                try:
                    aux.append(np.mean(aux_o[np.where(aux_o!=0)]) + aux[-1])
                except:
                    aux.append(np.mean(aux_o[np.where(aux_o!=0)]))
            self.all_pendings = self.all_pendings + list(reversed(aux))

        aux_list = []
        aux_list_2 = []
        aux_list_features = []
        for i in range(len(self.operations)):
            o = self.operations[i]
            for j in range(len(o)):
                t = o[j]
                if t!=0:
                    aux_list.append([i, j])
                    aux_list_2.append([j, i])
                    aux_list_features.append([t, t/np.sum(o), t/self.all_pendings[i], 0, 0])

        self.data['operation', 'exec', 'machine'].edge_index = torch.LongTensor(aux_list).T
        self.data['operation', 'exec', 'machine'].edge_attr = torch.Tensor(aux_list_features)

        self.data['machine', 'exec', 'operation'].edge_index = torch.LongTensor(aux_list_2).T
        self.data['machine', 'exec', 'operation'].edge_attr = torch.Tensor(aux_list_features)

        for i in range(self.num_jobs):
            o_index = 0
            for j in self.jobs[i]:
                self.data["operation"].x[j, 1] = self.all_pendings[j]
                aux = np.array(operations[j])
                o_index+=1

    def reset(self, sel_index = None):
        if sel_index is None:
            self.generate_instance(self.instances[self.current_instance])
            self.current_instance = (self.current_instance + 1)%len(self.instances)
        else:
            self.generate_instance(self.instances[sel_index])

        self.num_steps = 0
        self.change_machine = 0
        self.state = copy.deepcopy(self.data)

        self.job_start_machines = torch.empty((self.num_jobs,self.num_machines))
        self.current_job_proc = torch.zeros((self.num_jobs,self.num_machines))

        self.current_operations = [0]*self.num_jobs
        for j_id in range(len(self.jobs)):
            self.current_operations[j_id] = self.jobs[j_id][0]
        
        self.operations_ends = [0]*self.num_jobs
        self.machines_occupations = [0]*self.num_machines
        
        aux_list = []
        aux_list_features = []
        for j_id in range(len(self.jobs)):
            oper = self.operations[self.jobs[j_id][0]]
            for m in range(len(oper)):
                t = oper[m]
                if t!=0:
                    aux_list.append([m, j_id])
                    aux_list_features.append([t, t/np.sum(oper), t/np.sum(oper), 0, t])
                    self.job_start_machines[j_id,m] = 0
                    self.current_job_proc[j_id,m] = int(t)
                else:
                    self.job_start_machines[j_id,m] = 10000

        self.state['machine', 'exec', 'job'].edge_index = torch.LongTensor(aux_list).T
        self.state['machine', 'exec', 'job'].edge_attr = torch.Tensor(aux_list_features)

        self.calculate_next_state()
        self.calculate_mask()

        return self.state

    def calculate_mask(self):
        
        if self.mask_option == 0:
            mask_matrix = self.job_start_machines
        else:
            mask_matrix = self.job_start_machines + self.current_job_proc
        
        smallest = torch.unique(torch.topk(torch.flatten(mask_matrix), k = self.sel_k, largest = False, dim = 0).values)
        min_jobs = torch.tensor([])
        min_machines = torch.tensor([])

        for s in smallest:
            mj , mm = (mask_matrix == s).nonzero(as_tuple=True)
            min_machines = torch.concat([min_machines, mm])
            min_jobs = torch.concat([min_jobs, mj])

        pairs = torch.stack([min_machines, min_jobs]).T
        indexes = []
        for p in pairs:
            aux = self.state['machine', 'exec', 'job'].edge_index.T == p
            aux = np.logical_and(aux[:,0], aux[:,1])
            indexes = indexes + [i for i, val in enumerate(aux) if val==1]

        res = [True]*self.state['machine', 'exec', 'job'].edge_index.shape[1]
        for i in indexes:
            res[i] = False
        
        self.state['machine', 'exec', 'job'].mask = torch.BoolTensor(res)
    
    def calculate_next_state(self):

        self.state["machine"].x[:,2] = self.state["machine"].x[:,0] - torch.min(self.state["machine"].x[:,0])
        for j_id in range(len(self.jobs)):
            if int(self.state["job"].x[j_id,0])==0:
                o_id = self.current_operations[j_id]
                pj = self.state['operation', 'belongs', 'job'].edge_index[:,self.state['operation', 'belongs', 'job'].edge_index[1,:] == j_id]
                oper_id = pj[0,0]
                self.state["operation"].x[oper_id, 0] = 1
                #self.state["operation"].x[oper_id, 1] = self.all_pendings[o_id]
                self.state["job"].x[j_id, 1] = self.operations_ends[j_id]
                self.state["job"].x[j_id, 2] = pj.shape[1]
                self.state["job"].x[j_id, 3] = self.all_pendings[o_id]

        for m in range(len(self.state["machine"].x)):
            mask = self.state["operation", "exec", "machine"].edge_index[1,:] == m
            if mask.any().item():
                self.state["operation", "exec", "machine"].edge_attr[mask,4] = self.state["operation", "exec", "machine"].edge_attr[mask,0]/self.state["operation", "exec", "machine"].edge_attr[mask,0].max()
                mask = self.state["machine", "exec", "operation"].edge_index[0,:] == m
                self.state["machine", "exec", "operation"].edge_attr[mask,4] = self.state["machine", "exec", "operation"].edge_attr[mask,0]/self.state["machine", "exec", "operation"].edge_attr[mask,0].max()

            mask = self.state["machine", "exec", "job"].edge_index[0,:] == m
            if mask.any().item():
                self.state["machine", "exec", "job"].edge_attr[mask,4] = self.state["machine", "exec", "job"].edge_attr[mask,0]/self.state["machine", "exec", "job"].edge_attr[mask,0].max()
                #self.state["machine"].x[m, 7] = self.state["machine", "exec", "job"].edge_attr[mask,0].shape[0]                   
    def step(self, action):

        self.num_steps+=1       
        
        action = self.state['machine', 'exec', 'job'].edge_index[:,action]

        sel_job = action[1]
        sel_mach = action[0]

        mask = self.state['machine', 'exec', 'job'].edge_index[1,:] != sel_job
        self.state['machine', 'exec', 'job'].edge_index = self.state['machine', 'exec', 'job'].edge_index[:,mask]
        self.state['machine', 'exec', 'job'].edge_attr = self.state['machine', 'exec', 'job'].edge_attr[mask]
        
        prev_ms = float(torch.max(self.state["machine"].x[:,0]))
        o_id = self.current_operations[sel_job]

        start_time = max(self.state["machine"].x[sel_mach,0], self.operations_ends[sel_job])
        proc_time  = self.operations[o_id][sel_mach]

        final_time =  start_time + proc_time
        self.state["machine"].x[sel_mach, 0] = final_time

        self.job_start_machines[self.job_start_machines[:,sel_mach] <  final_time, sel_mach] =  final_time
        self.machines_occupations[sel_mach] += proc_time
        self.state["machine"].x[sel_mach, 1] = self.machines_occupations[sel_mach]/final_time

        self.operations_ends[sel_job] = final_time
        self.job_start_machines[sel_job,:] = 10000
        self.current_job_proc[sel_job, :] = 0

        if (int(self.current_operations[sel_job])) == self.jobs[sel_job][-1]:
            self.state["job"].x[sel_job,:]=0
            self.state["job"].x[sel_job,0]=1

            self.state['job', 'listens', 'job'].edge_index = self.state['job', 'listens', 'job'].edge_index[:, self.state['job', 'listens', 'job'].edge_index[0,:] != sel_job]

        else:
            self.current_operations[sel_job]+=1
            aux_list = []
            aux_list_features = []
            oper= np.array(self.operations[self.current_operations[sel_job]])
            total_gap = 0
            for m in range(len(oper)):
                t = oper[m]
                if t!=0:
                    calcu = t + max(self.operations_ends[sel_job] - self.state["machine"].x[m, 0],0)
                    total_gap += calcu
                    aux_list.append([m, sel_job])
                    aux_list_features.append([calcu, t/np.sum(oper) , t + max(self.operations_ends[sel_job], self.state["machine"].x[m, 0]) ])
                    self.job_start_machines[sel_job,m] = max(self.operations_ends[sel_job], self.state["machine"].x[m, 0])
                    self.current_job_proc[sel_job,m] = t
            
            for l in aux_list_features:
                l.append(l[0]/total_gap)
                l.append(0)

            self.state['machine', 'exec', 'job'].edge_index = torch.concat([self.state['machine', 'exec', 'job'].edge_index, torch.LongTensor(aux_list).T] , dim=1)
            self.state['machine', 'exec', 'job'].edge_attr = torch.concat([self.state['machine', 'exec', 'job'].edge_attr, torch.Tensor(aux_list_features)])
        
        reward = prev_ms - float(torch.max(self.state["machine"].x[:,0])) 
        
        if torch.all(self.state["job"].x[:,0]==1):
            self.mk = round(float(torch.max(self.state["machine"].x[:,0])),2)
            return self.state, reward , True, {"current_machine": sel_mach}
    
        oper_id = self.state['operation', 'belongs', 'job'].edge_index[:,self.state['operation', 'belongs', 'job'].edge_index[1,:] == sel_job][0,0]
        self.state['operation', 'belongs', 'job'].edge_index = self.state['operation','belongs', 'job'].edge_index[:,self.state['operation', 'belongs', 'job'].edge_index[0,:] != oper_id]
        self.state['operation', 'prec', 'operation'].edge_index = self.state['operation', 'prec', 'operation'].edge_index[:,self.state['operation', 'prec', 'operation'].edge_index[1,:] != oper_id]

        mask = self.state['operation', 'exec', 'machine'].edge_index[0,:] != oper_id
        self.state['operation', 'exec', 'machine'].edge_index = self.state['operation', 'exec', 'machine'].edge_index[:,mask]
        self.state['operation', 'exec', 'machine'].edge_attr = self.state['operation', 'exec', 'machine'].edge_attr[mask]

        mask = self.state['machine', 'exec', 'operation'].edge_index[1,:] != oper_id
        self.state['machine', 'exec', 'operation'].edge_index = self.state['machine', 'exec', 'operation'].edge_index[:,mask]
        self.state['machine', 'exec', 'operation'].edge_attr = self.state['machine', 'exec', 'operation'].edge_attr[mask]
        self.state = T.RemoveIsolatedNodes()(self.state)
        
        self.calculate_next_state()
        self.calculate_mask()

        done = False
        total_reward = reward
        if len(self.state['machine', 'exec', 'job'].mask) - sum(self.state['machine', 'exec', 'job'].mask)==1:
            self.state, reward , done, _ = self.step(self.sample())
            total_reward +=reward

        return self.state, total_reward , done, {"current_machine": sel_mach}

    def sample(self):
        return random.choice([i for i in range(len(self.state['machine', 'exec', 'job'].mask)) if not self.state['machine', 'exec', 'job'].mask[i]])
    
    def normalize_state(self, state):
        state = copy.deepcopy(state)
        for i in range(state["job"].x.shape[1]):
            state["job"].x[:,i] = (2*(state["job"].x[:,i] - state["job"].x[:,i].min())/(state["job"].x[:,i].max() - state["job"].x[:,i].min() + 1e-7 )-1).float()
        
        for i in range(state["operation"].x.shape[1]):
            state["operation"].x[:,i] = (2*(state["operation"].x[:,i] - state["operation"].x[:,i].min())/(state["operation"].x[:,i].max() - state["operation"].x[:,i].min() + 1e-7 )-1).float()
        
        for i in range(state["machine"].x.shape[1]):
            state["machine"].x[:,i] = (2*(state["machine"].x[:,i] - state["machine"].x[:,i].min())/(state["machine"].x[:,i].max() - state["machine"].x[:,i].min() + 1e-7 )-1).float()

        state[('operation', 'exec', 'machine')].edge_attr = (2*(state[('operation', 'exec', 'machine')].edge_attr -  state[('operation', 'exec', 'machine')].edge_attr.min())/(state[('operation', 'exec', 'machine')].edge_attr.max() - state[('operation', 'exec', 'machine')].edge_attr.min() + 1e-7 )-1).float()
        state[('machine', 'exec', 'operation')].edge_attr = (2*(state[('machine', 'exec', 'operation')].edge_attr -  state[('machine', 'exec', 'operation')].edge_attr.min())/(state[('machine', 'exec', 'operation')].edge_attr.max() - state['machine', 'exec', 'operation'].edge_attr.min() + 1e-7 )-1).float()
        state[('machine', 'exec', 'job')].edge_attr = (2*(state[('machine', 'exec', 'job')].edge_attr - state[('machine', 'exec', 'job')].edge_attr.min())/(state[('machine', 'exec', 'job')].edge_attr.max() - state[('machine', 'exec', 'job')].edge_attr.min() + 1e-7 )-1).float()
        return state
    
