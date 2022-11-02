#!/bin/python3

# Implementation of Siminet algorithm in Python

import networkx as nx
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from functools import wraps

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

def annotate_merge(fn):
    @wraps(fn)
    def wrapped(gcmp, gcmp_pos, gref, gref_pos, *args, **kwargs):
        merge_equivalent(gcmp, gcmp_pos)
        annotate_graph(gcmp, gcmp_pos)
        
        merge_equivalent(gref, gref_pos)
        annotate_graph(gref, gref_pos)
        
        return distance(gcmp, gref, *args, **kwargs)
    return wrapped
            

def max_cost_score(node_score, edge_weight_score, edge_dist_score, gcmp, gref, eps, delta, tau):
    """
    Scoring function that normalizes the sum of the node score and edge score with
    the maximum possible such score  -- which is the sum of maximum insertion cost and maximum deletion cost
    for both nodes and edges.
    """
    #print(max_cost_score.__name__)
    sub_rad = 2*eps
    # Maximum node costs
    max_node_ins_cost = sub_rad*len(gref.nodes)
    max_node_del_cost = sub_rad*len(gcmp.nodes)

    # Maximum edge costs
    max_edge_ins_cost = sum(attrs["weight"] for (_,_,attrs) in gref.edges(data=True))
    max_edge_del_cost = sum(attrs["weight"] for (_,_,attrs) in gcmp.edges(data=True))
    #max_edge_ins_cost = 2*(delta+tau)*len(gref.edges)
    #max_edge_del_cost = 2*(delta+tau)*len(gcmp.edges)


    max_score   = max_node_ins_cost + max_node_del_cost + max_edge_ins_cost + max_edge_del_cost
    given_score = node_score + edge_dist_score + edge_weight_score

    #print(f"{max_score=}")
    return given_score / max_score

#@annotate_merge
def distance(gcmp, gref, eps, delta, tau, sub_rad=None , scoring_func=None, transform=False, ins_cost = None):
    """
    Intakes two graphs, gcmp and gref, and computes the node and edge distances b/w them as per the Siminet algorithm.
    The graph is expected to be labeled, with the nodes having an attribute 'position' (corresponds to spatial position)
    and the edges having an attribute 'weight'. The computed node and edge distances, as well as the two graphs,
    will be given to a scoring function, which will (usually) yield some scalar -- this will be returned as the final result. 
    
    The scoring function is a function which intakes the node score, the edge score, the comparision graph (gcmp) 
    and the reference graph (gref). If scoring_func is set to None, then the returned value is a tuple of the node score and edge score.
    
    The transform boolean flag will transform a copy of gcmp during the course of the function if set to true 
    (for testing purposes).
    """
    
    #eps = 7*eps
    
    if scoring_func is None: # scoring function, using the node/edge scores and the two graphs
        scoring_func = lambda n,ew, ed, gc, gr, eps, delta, tau: (n,ew, ed) # default just returns back the node and edge scores

    if sub_rad is None:
        sub_rad = 2*eps
        #sub_rad = 5*eps
    
    assert eps < sub_rad
        
    copy = nx.empty_graph()
    
    if transform: 
        copy = deepcopy(gcmp) # ensures that we don't mutate what was passed in
    
    dist = lambda p,q: np.linalg.norm(p[1]["position"] - q[1]["position"]) # compute the Euclidean distance b/w nodes

    equivalency_mapping = dict() # maps nodes in Gref to nodes in Gcmp, representing equality b/w them
    counterpart_nodes   = set() # set of all nodes in Gref with counterparts in Gcmp (through substitution or equality)
    
    node_score = 0
    if ins_cost is None:
        ins_cost = sub_rad
    
    freq_table = {'equivalency'  : 0,
                  'substitution' : 0,
                  'deletion'     : 0,
                 }
    
    avg_del_dist = 0
    distances = []
    
    valid_cand = lambda nde: nde[0] not in counterpart_nodes # condition to ensure that candidate node is new, hasn't been seen before

    # NODE SCORE
    
    gcmp_attrs_sorted = sorted(gcmp.nodes(data=True), key=lambda n: n[0])
    gref_attrs_sorted = sorted(gref.nodes(data=True), key=lambda n: n[0])
        
    for n in gcmp_attrs_sorted:
        
        #valid_cand = lambda nde: nde[0] not in counterpart_nodes # condition to ensure that candidate node is new, hasn't been seen before
        closest = min(filter(valid_cand, gref_attrs_sorted), 
                      key     = lambda m: dist(m, n),
                      default = (None, {"position": np.array([np.inf,np.inf,np.inf])})) 
        # if gref is empty, default value returned is node at infinity
       
        d = dist(n, closest)
        
        #if d != np.inf:
        #    distances.append(d)

        if d <= eps: # Equivalency
            #print("Equivalency")
            freq_table['equivalency'] += 1
            equivalency_mapping[closest[0]] = n[0]
            counterpart_nodes.add(closest[0])
            
            if transform:
                copy.nodes[n[0]]["position"] = closest[1]["position"]
        
        elif d <= sub_rad: # Substitution
            #print(f"Substitution, {d}")
            freq_table['substitution'] += 1
            #equivalency_mapping[closest[0]] = n[0]
            # equivalency_mapping[n] = closest
            counterpart_nodes.add(closest[0])
            node_score += d
            if transform:
                copy.nodes[n[0]]["position"] = closest[1]["position"]
        
        else: # Deletion
            #print("Deletion")
            freq_table['deletion'] += 1
            avg_del_dist += d
            node_score += ins_cost
            if transform:
                copy.remove_node(n[0])
        #print(f"\t{d=}, {eps=}, {sub_rad=}")

    not_found   = gref.nodes - counterpart_nodes # nodes in Gref that had no counterpart (equivalency or substitution) in Gcmp
    #print(f"Insertion: {len(not_found)}")
    freq_table['insertion'] = len(not_found)
    node_score += ins_cost * len(not_found) # total insertion cost for nodes not found

    if transform: # Node Insertion, if we are transforming the copy of Gcmp
        for n in not_found:
            copy.add_node(n, **gref.nodes[n])

    
    # EDGE SCORE
    
    #edge_score = 0
    edge_weight_score = 0 # number of fibers
    edge_dist_score   = 0 # length of edge (Euclidean distance b/w nodes)
    
    counterpart_edges = set()
        
    #print("edge score")
    for e in gref.edges(data=True):
        n1, n2, ref_data = e
        wt               = 0
        edist            = 0
        add_edge         = False
        
        if n1 in equivalency_mapping and n2 in equivalency_mapping:
            pce = (equivalency_mapping[n1], equivalency_mapping[n2]) # finds (potential) corresponding edge based on equivalency mapping

            if pce in gcmp.edges:
                wt = gcmp.edges[pce]["weight"]
                edist = gcmp.edges[pce]["distance"]
                counterpart_edges.add(pce)
                
            else:
                add_edge = True
        else:
            add_edge = True

        raw_wt_diff   = abs(ref_data["weight"] - wt)
        raw_dist_diff = abs(ref_data["distance"] - edist)
        
        edge_weight_score += raw_wt_diff if raw_wt_diff > (tau/len(gref.edges)) else 0
        edge_dist_score   += raw_dist_diff if raw_dist_diff > delta else 0
        
        #edge_weight_score += np.median([delta, 10*delta, raw_wt_diff])
        #edge_dist_score   += np.median([tau, 10*tau, raw_dist_diff])
        #print(edge_weight_score, edge_dist_score)
        
        #if sc < delta:
        #    sc = 0
        #elif sc > 2*delta:
        #    sc = 2*delta
            
        #edge_weight_score += sc
        
        if transform and add_edge: # Edge Insertion, if we are transforming the copy of Gcmp
            copy.add_edge(n1, n2, **ref_data)

    lone_edges = gcmp.edges - counterpart_edges # Edges with no counterpart in Gref
    weight_deletion_score = sum(gcmp.edges[e]["weight"] for e in lone_edges)
    dist_deletion_score   = sum(gcmp.edges[e]["distance"] for e in lone_edges)
    # Deletion score is computed by summing the weights of lone edges

    edge_weight_score += weight_deletion_score
    edge_dist_score   += dist_deletion_score

    #print(f"\t{freq_table=}")
    
    #if freq_table['deletion']:
    #    avg_del_dist /= freq_table['deletion']
    #    print(f"{avg_del_dist=}, {avg_del_dist / eps}")
    
    #plt.figure()
    #plt.hist(np.array(distances), bins=30)
    #plt.show()
        
    #final_score = scoring_func(node_score, edge_weight_score, gcmp, gref, eps, delta, tau)
    final_score = scoring_func(node_score, edge_weight_score, edge_dist_score, gcmp, gref, eps, delta, tau)
    #print(f"siminet {final_score=}")
        
    if transform:
        #return scoring_func(node_score, edge_weight_score, gcmp, gref, sub_rad), copy
        return final_score, copy
    
    #return scoring_func(node_score, edge_weight_score, gcmp, gref, sub_rad)
    return final_score