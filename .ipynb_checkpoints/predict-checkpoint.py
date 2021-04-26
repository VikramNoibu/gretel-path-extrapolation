#!/usr/bin/env python
# coding: utf-8

# In[2]:


from main import *
from tqdm import tqdm


# In[3]:


domain = 'pollini'
model_path = "data/chkpt/"+domain+"/0001.pt"

config_path = "config/" + domain
config = Config()
config.load_from_file(config_path)
TRAJ_MAX_LEN=99
graph, trajectories, pairwise_node_features, _ = load_data(config)
model = create_model(graph, pairwise_node_features, config)
model = model.to(config.device)
model.load_state_dict(torch.load(model_path)["model_state_dict"])
model.eval()
graph = graph.to(config.device)


# In[4]:


TRAJ_MAX_LEN=99


# In[5]:


def create_evaluator():
    return Evaluator(graph.n_node, given_as_target=None, siblings_nodes=None,config=config,)
def predict(model, graph, trajectories, config, pairwise_features=None):
    preds= []
    with torch.no_grad():
        for trajectory_idx in tqdm(range(len(trajectories))):
            observations = trajectories[trajectory_idx]
            number_steps = None
            if config.rw_edge_weight_see_number_step or config.rw_expected_steps:
                if config.use_shortest_path_distance:
                    number_steps = (
                        trajectories.leg_shortest_lengths(
                            trajectory_idx).float() * 1.1
                    ).long()
                else:
                    number_steps = trajectories.leg_lengths(trajectory_idx)

            observed, starts, targets = generate_masks(
                trajectory_length=observations.shape[0],
                number_observations=config.number_observations,
                predict=config.target_prediction,
                with_interpolation=config.with_interpolation,
                device=config.device,
            )

            diffusion_graph = (
                graph if not config.diffusion_self_loops else graph.add_self_loops()
            )

            predictions, _, rw_weights = model(
                observations,
                graph,
                diffusion_graph,
                observed=observed,
                starts=starts,
                targets=targets,
                pairwise_node_features=pairwise_features,
                number_steps=number_steps,
            )
            preds.append(predictions.argmax(dim=1))
        
    return preds


# In[6]:


valid_trajectories = trajectories[2].to(config.device)
evaluate(model, graph,
        valid_trajectories, pairwise_node_features,
        create_evaluator, dataset="EVAL")


# In[7]:


edges2ids = {(val[0].item(), val[1].item()):idd for idd, val in enumerate(zip(graph.senders,graph.receivers))}


# In[8]:


def path2traj(paths, num_nodes):
    """converts a simple nodewise list of paths to trajectory format required by gretel"""
    
    if not isinstance(paths[0],list):
        """if there is just one path in traj"""
        paths=[paths]
    
    total_observations = sum([len(path) for path in paths])
    lengths=[]
    indices = torch.zeros(total_observations, 1, dtype=torch.long)
    num_observations = 0
    
    for path in paths:
        length = len(path)
        assert length > config.min_trajectory_length, "Min length constraint not satisfied"
        lengths.append(length) 
        for i,node in enumerate(path):
            indices[i+num_observations, 0] = node
        num_observations+=length  
    
    weights = torch.ones(num_observations, 1)
    
    num_paths = num_observations - len(paths)
    max_path_length= TRAJ_MAX_LEN
    trails = torch.zeros([num_paths, max_path_length], dtype=torch.long) - 1
    for i, path in enumerate(paths):
        edges = [edges2ids[(e1,e2)] for (e1,e2) in zip(path[:-1],path[1:])]
        trails[i, :len(edges)] = torch.tensor(edges)
    return Trajectories(weights, indices, num_nodes, torch.tensor(lengths), trails)


# In[24]:


paths = [[1,3,0,1,0, 2,0,4, 5, 6, 5, 1], [0, 1, 3,3, 0, 0, 1, 0 ,5, 6, 5,1]]
trajs = path2traj(paths, graph.n_node)
trajs = trajs.to(config.device)


# In[25]:


preds = predict(model, graph, trajs, config)


# In[26]:


preds


# ##
# + Train on organic paths 
# + Test on error paths 
#     - file to paths
#     - test various algorithms
#     - metrics
#     

# In[32]:


obs, starts, targets = generate_masks(10, 5)


# In[38]:


obs, starts, targets


# In[35]:


trajs[0][targets]


# In[37]:


trajs[0]


# In[ ]:




