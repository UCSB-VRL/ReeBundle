#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dipy.segment.bundles as dsb
import nibabel as nib
import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram
import nibabel as nib
import networkx as nx
import matplotlib.pyplot as plt
import robustReebConstruction_avg_loc_qb as rc
import os
import pickle
import networkx as nx
import visualization as vis
from mpl_toolkits.mplot3d import Axes3D

#read streamlines and create combined_streamlines
combined_streamline = []
for i in range (1,10):
    try:
        print("sess-",i)
        trkpathI = "/media/hdd2/shailja/Crash/sphere1brush6/sub-1145h_ses-"+str(i)+"_DSI_mc.src.gz.odf8.f5rec.gqi.1.25.cc.trk.gz"
        p_streamlines =  nib.streamlines.load(trkpathI)
        print("n = ",len(p_streamlines.streamlines))
        count = 0
        streamlines = p_streamlines.streamlines
        for i in range(len(streamlines)):
            count+=len(streamlines[i])
            combined_streamline.append(streamlines[i])
        print("N = ",count)
    except:
        pass


# In[2]:


def graph_vis(G, node_loc, streamlines):
    # 3d spring layout
    pos = node_loc
    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Create the 3D figure
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, ec="w",label = None)
    for i in range(len(streamlines)):
        xdata = []
        ydata = []
        zdata = []
        for j in streamlines[i]:
            xdata.append(j[0])
            ydata.append(j[1])
            zdata.append(j[2])
#         ax.plot3D(xdata,ydata,zdata,color= '#bfbfbf', lw = 2);
        ax.plot3D(xdata,ydata,zdata,color= '#eb7a30', lw = 2, alpha = 0.2);
    # Plot the nodes
    ax.scatter(*node_xyz.T, s=400, ec="w", color = 'r', zorder=100)
    edge_labels = nx.get_edge_attributes(G, "weight")
    # Plot the edges
    weight_labels = list(edge_labels.values())
    count = 0
    for vizedge in edge_xyz:
        wt = weight_labels[count]/max(weight_labels)*5
#         if wt == 10:
#             ax.plot(*vizedge.T, color='g',
#                     lw = wt,
#                     zorder = 50,
#                    label = str(weight_labels[count]))
#         else:
        ax.plot(*vizedge.T, color='#000000',
                lw = wt,
                zorder = 50,
               label = str(weight_labels[count]))
        count+=1


    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")


    _format_axes(ax)
    fig.tight_layout()
    plt.axis("off")
    plt.savefig(trkpathI.split("_")[1]+".png", dpi=300)
#     plt.legend()




def compute_metric(params):
    eps, delta, tau = params
    H, node_loc,cluster_edge = rc.constructRobustReeb(streamlines, eps, delta, tau)
    if not nx.is_empty(H) and nx.is_connected(H):
        merge_equivalent(H1, node_loc1)
        annotate_graph(H1, node_loc1)
        merge_equivalent(H, node_loc)
        annotate_graph(H, node_loc)
        
        dist = partial(sn.distance, eps=2.5, delta=3, tau=5, scoring_func=my_scoring_fn)
#         dist_ref = partial(sn.distance, eps=eps, delta=delta, tau=tau, scoring_func=my_scoring_fn)
        #dist = partial(sn.distance, eq_rad=eps, scoring_func=sn.max_cost_score)
#         dist = partial(sn.distance, eq_rad=eps, scoring_func=lambda ns, es, gc, gr, rad: ns+es)
#         dist = netrd.distance.NetLSD()
        return 0.5*dist(H,H1)
    else:
        return np.inf
def my_scoring_fn(node_score, edge_weight_score, edge_dist_score, gcmp, gref, eps, delta, tau):
    return node_score/len(gref.nodes()) # throw away everything else    
    
import netrd
import siminet as sn
from functools import partial

def annotate_graph(graph, node_positions):
    for e in graph.edges:
        n1, n2 = e
        pos1, pos2 = np.array(node_positions[n1]), np.array(node_positions[n2])
        graph.nodes[n1]["position"] = pos1
        graph.nodes[n2]["position"] = pos2
        graph.edges[e]["distance"] = np.linalg.norm(pos1 - pos2)
        
def merge_equivalent(graph, node_annotations):
    """
    Intakes a graph and its associated node annotations where some nodes may have the same annotation (spatial position). 
    Those equivalent nodes will be merged into the same node, and edges involving these equivalent nodes will be inherited 
    by the final node.
    """
    
    equivalences = dict()
    
    for pos, node in node_annotations.items():
        if pos not in equivalences:
            equivalences[pos] = []
        
        equivalences[pos].append(node)
        
    for eq_group in equivalences.values():
        if len(eq_group) == 1: # nothing to merge
            continue
            
        head, tail = eq_group[0], eq_group[1:]
        for n in tail:
            nx.contracted_nodes(graph, head, n, copy=False)   


# In[6]:
trkpathI = "/media/hdd2/shailja/Crash/sphere1brush6/sub-3058s_ses-1_DSI_mc.src.gz.odf8.f5rec.gqi.1.25.cc.trk.gz"    
p_streamlines =  nib.streamlines.load(trkpathI)
print("n = ",len(p_streamlines.streamlines))
count = 0
streamlines = p_streamlines.streamlines
eps = 2.5

delta = 3
tau = 5
H1, node_loc1, cluster_edge= rc.constructRobustReeb(streamlines, eps, delta, tau)
vis.visualizeReebNodes(streamlines, H1, node_loc1)
graph_vis(H1, node_loc1, streamlines)

#Bayesian
##
# prior = {'eps':['gaussian', eps0, eps_var, 'positive', 'integer'], 'delta':['uniform', min, max]}
# params_dict = {'eps':eps_val, 'delta':delta_val}
cost_progress = []
cost_params = []
def check_prior(prior, params_dict):
    lp = 0.0
    for key,value in params_dict.items():
        if 'positive' in prior[key] and value  < 0:
            return np.inf
        if 'integer' in prior[key] and type(value) is not int:
            return np.inf
        prior_type = prior[key][0]
        if prior_type == 'uniform':
            lp += uniform_prior(prior, key, value)
        elif prior_type == 'gaussian':
            lp += gaussian_prior(prior, key, value)
    return lp

def uniform_prior(prior, param_name, param_value):
    '''
    Check if given param_value is valid according to the prior distribution.
    Returns np.Inf if the param_value is outside the prior range and 0.0 if it is inside. 
    param_name is used to look for the parameter in the prior dictionary.
    '''
    prior_dict = prior
    if prior_dict is None:
        raise ValueError('No prior found')
    lower_bound = prior_dict[param_name][1]
    upper_bound = prior_dict[param_name][2]
    if param_value > upper_bound or param_value < lower_bound:
        return np.inf
    else:
        return np.log( 1/(upper_bound - lower_bound) )

def gaussian_prior(prior, param_name, param_value):
    '''
    Check if given param_value is valid according to the prior distribution.
    Returns the log prior probability or np.Inf if the param_value is invalid. 
    '''
    prior_dict = prior
    if prior_dict is None:
        raise ValueError('No prior found')
    mu = prior_dict[param_name][1]
    sigma = prior_dict[param_name][2]
    if sigma < 0:
        raise ValueError('The standard deviation must be positive.')
    # Using probability density function for normal distribution
    # Using scipy.stats.norm has overhead that affects speed up to 2x
    prob = 1/(np.sqrt(2*np.pi) * sigma) * np.exp(-0.5*(param_value - mu)**2/sigma**2)
    if prob < 0:
        warnings.warn('Probability less than 0 while checking Gaussian prior! Current parameter name and value: {0}:{1}.'.format(param_name, param_value))
        return np.inf
    else:
        return np.log(prob)

# params is a list of values at each step
def get_likelihood_function(params, prior, params_to_estimate):
    params_dict = {}
    for key, p in zip(params_to_estimate, params):
        params_dict[key] = p
    # Set the params (list of values) to your code and compute cost
#         eps, delta, tau = params
    # Check prior
    lp = 0
    lp = check_prior(prior, params_dict)
    if not np.isfinite(lp):
        return -np.inf
    #apply cost function
    # compute d - d_ideal, normed and log
#         LL_det_cost = self.LL_det.py_log_likelihood()
#         ln_prob = lp + LL_det_cost
    ll_cost = -1*compute_metric(params)
    cost_params.append(params)
    cost_progress.append(-1*ll_cost)
    ln_prob = lp + ll_cost
    return ln_prob
        


# In[7]:


idx =9 
print(idx)
trkpathI = "/media/hdd2/shailja/Crash/sphere1brush6/sub-3058s_ses-"+ str(idx)+"_DSI_mc.src.gz.odf8.f5rec.gqi.1.25.cc.trk.gz" 
p_streamlines =  nib.streamlines.load(trkpathI)
streamlines = p_streamlines.streamlines
import emcee
params_to_estimate = ['eps', 'delta', 'tau']
prior = {'eps':['gaussian', 2.5, 4, 'positive'], 
         'delta':['gaussian',3, 4, 'positive'],
         'tau':['gaussian', 5, 4, 'positive']}
init_values = [2.5,3,5] # Set

ndim = len(init_values)
nwalkers = 10
nsteps = 1000
pos = np.array(init_values) + 1e-4 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, get_likelihood_function, args=(prior, params_to_estimate))
sampler.run_mcmc(pos, nsteps, progress=True);
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
# labels = ["m", "b", "log(f)"]
labels = ["$\epsilon$", "$\\alpha$", "$\delta$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
import corner
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)
fig = corner.corner(
    flat_samples, labels=labels, levels = (0.75, ), truths=[init_values[0], init_values[1], init_values[2]]
);
fig.savefig('3058_3058scornerplotCC2.5_3_5_'+str(idx)+'.svg')




