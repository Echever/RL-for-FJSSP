import torch
import torch.nn as nn
from torch_geometric.nn import  GATv2Conv, Linear, to_hetero
from torch_geometric.data import HeteroData
from torch.distributions import Categorical
from torch_geometric.nn import global_mean_pool
import copy
from torch_geometric.loader import DataLoader
import random

device = torch.device('cpu')

if(torch.cuda.is_available()) and random.random()<1: 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    #print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    pass
    #print("Device set to : cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers = 2, heads = 2):
        super().__init__()
        self.lin1 = Linear(-1, 8)
        #self.lin2 = Linear(-1, out_channels)
        self.s = torch.nn.Softmax(dim=0)
        self.tanh = nn.Tanh()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GATv2Conv(-1, hidden_channels, add_self_loops=False, edge_dim=5, heads = heads)
            self.convs.append(conv)

    def forward(self, x, edge_index, edge_attr_dict):
        x = self.lin1(x)
        x = self.tanh(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr_dict)
        x = self.tanh(x)
        #x = self.lin2(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, actor, num_layers = 2, heads = 3):
        super().__init__()
        self.actor = actor
        self.gnn = GAT(hidden_channels, out_channels, num_layers=num_layers, heads=heads)
        self.gnn = to_hetero(self.gnn, metadata=metadata, aggr='mean')
        self.lin3 = Linear(-1, 1)
        self.lin4 = Linear(-1, hidden_channels)
        self.lin5 = Linear(-1, hidden_channels)
        self.lin6 = Linear(-1, 1)
        self.tanh = nn.Tanh()

    def forward(self, data: HeteroData):
        res = self.gnn(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        if self.actor:
            x_src, x_dst = res['machine'][data.edge_index_dict[('machine','exec','job')][0]], res['job'][data.edge_index_dict['machine','exec','job'][1]]
            edge_feat = torch.cat([x_src,  data.edge_attr_dict[('machine','exec','job')], x_dst], dim=-1)

            # if "batch" not in data["job"]:
            #     expanded_vector = res["nglobal"][0].unsqueeze(0).expand(edge_feat.size(0), -1)
            #     edge_feat = torch.cat((edge_feat, expanded_vector), dim=1)
            # else:
            #     nglobals = []
            #     for ind in data['machine','exec','job'].edge_index.T:
            #         nglobals.append(res["nglobal"][data["job"].batch[ind[1]]])
            #     nglobals = torch.stack(nglobals)
            #     edge_feat = torch.cat((edge_feat, nglobals), dim=1)

            #res = self.lin4(edge_feat)
            #res = self.tanh(res)
            res = self.lin3(edge_feat)
        else:
            # res = self.lin5(res["job"])
            # res = self.tanh(res)

            res = self.lin6(res["job"])
            res = self.tanh(res)

        return res
        

class ActorCritic(nn.Module):
    def __init__(self,metadata, hidden_channels=128, num_layers=2, heads = 3):
        super(ActorCritic, self).__init__()
        self.actor = Model(hidden_channels, 32, metadata, True, num_layers, heads)
        self.critic = Model(hidden_channels, 32, metadata, False, num_layers, heads)
        self.metadata = metadata
        self.soft = torch.nn.Softmax(dim=0)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, sample, num):
        action_probs = self.actor(state).T[0]
        action_probs[state[('machine','exec','job')].mask] = float("-inf")
        action_probs = self.soft(action_probs)
                
        dist = Categorical(action_probs)
        
        if sample == 0:
            action = dist.sample()
        else:
            #if sample == 2:
            #    print(action_probs)
            action = torch.argmax(action_probs)

        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):
        action_probs = torch.Tensor([])
        res = self.actor(state)
        action_logprobs = []
        dist_entropy = []

        row, col = state[('machine', 'exec', 'job')].edge_index
        batch_index = state["machine"].batch[row]

        for i in range(state["job"].batch[-1]+1):
            action_probs = res[batch_index==i].T[0]
            action_probs[state[('machine','exec','job')].mask[batch_index==i]] = float("-inf")
            action_probs = self.soft(action_probs)
            dist = Categorical(action_probs)
            action_logprobs.append(dist.log_prob(action[i]))
            dist_entropy.append(dist.entropy())
        
        action_logprobs = torch.stack(action_logprobs)
        dist_entropy =  torch.stack(dist_entropy)

        state_values = self.critic(state)
        state_values = global_mean_pool(state_values, state["job"].batch)
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, lr_actor, lr_critic, gamma, K_epochs, eps_clip, env, metadata, hidden_channels=128, num_layers = 2,  batch_size=64, heads = 3):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.env = env
        self.metadata = metadata
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(metadata,hidden_channels, num_layers, heads).to(device)

        #self.policy.actor = self.policy.actor.to(device)
        #self.policy.critic = self.policy.critic.to(device)

        self.policy_old = ActorCritic(metadata,hidden_channels, num_layers, heads).to(device)
        self.policy_old = ActorCritic(metadata,hidden_channels, num_layers, heads).to('cpu')

        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        aux_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(aux_old.state_dict())
        self.policy_old.actor = self.policy_old.actor.to(device)
        self.policy_old.critic = self.policy_old.critic.to(device)
        
        self.MseLoss = nn.MSELoss()

        self.batch_size = batch_size

    def select_action(self, state, sample, num):
        with torch.no_grad():
            state = self.env.normalize_state(state)
            state = state.to(device)
            action, action_logprob = self.policy_old.act(state, sample, num)
        
        if sample!=2:
            #self.buffer.states.append(state)
            self.buffer.states.append(copy.deepcopy(state))
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

        return action
        #return action.item()

    def update(self):
        # Monte Carlo estimate of returns

        #print("Start update  : ", datetime.now().replace(microsecond=0) - start_time)

        all_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            all_rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        all_rewards = torch.tensor(all_rewards, dtype=torch.float32).to(device)
        if len(all_rewards)>1:
            all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-7)
            #all_rewards = (2*(all_rewards + 2200)/(2200 + 1e-7 )-1).float()

        # convert list to tensor
        batch_size = self.batch_size
        loader = DataLoader(self.buffer.states, batch_size=batch_size)
        all_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        all_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        all_losses = 0
        all_losses_cri = 0

        #print(all_rewards)
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            i=0
            for data in loader:
                data = data.to(device)
                old_actions = all_actions[i*batch_size: (i+1)*batch_size]
                old_logprobs = all_logprobs[i*batch_size: (i+1)*batch_size]
                rewards = all_rewards[i*batch_size: (i+1)*batch_size]
                logprobs, state_values, dist_entropy = self.policy.evaluate(data, old_actions)
                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())
                # Finding Surrogate Loss
                advantages = rewards - state_values.detach()

                all_losses_cri += 0.5*self.MseLoss(state_values, rewards).mean()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                # final loss of clipped objective PPO

                #print("-torch.min(surr1, surr2)", -torch.min(surr1, surr2))
                #print("- 0.01*dist_entropy", - 0.01*dist_entropy)
                #print("0.5*self.MseLoss(state_values, rewards)", 0.5*self.MseLoss(state_values, rewards))
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
                                    
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                i+=1
                all_losses+=loss.mean()

            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        #print("End update  : ", datetime.now().replace(microsecond=0) - start_time)

        return float(all_losses)/(i*self.K_epochs), float(all_losses_cri)/(i*self.K_epochs)
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))