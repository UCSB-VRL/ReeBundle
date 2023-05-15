import os
import nibabel as nib
import numpy as np
import networkx as nx
import pickle
from events import *
import numpy as np
import nibabel as nib
import pickle
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import Streamlines,set_number_of_points


def distance(p1, p2):
    """
    Computes Euclidean Distance
    Input: Two 3D points
    Output: Euclidean Distance betweenthe points
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
      
def constructRobustReeb(streamlines, eps, alpha, delta):
    """
    Reeb Graph Computation
    Input: Streamline file and teh parameters
    Output: Reeb Graph and Node location map assigning 3D coordinates to each node in the Reeb graph
    """
    cluster_map = {}
    threshold = 1.5
#     feature = ResampleFeature(nb_points=40)
#     metric = AveragePointwiseEuclideanMetric(feature=feature) 
#     qb = QuickBundles(threshold, metric=metric)
    qb = QuickBundles(threshold)
    clusters = qb.cluster(streamlines)
    centroid_trk = []
    for i in range(len(clusters)):
        centroid = []
        cluster_map[len(centroid_trk)] = len(clusters[i])
        for j in range(len(clusters[i].centroid)):
            centroid.append([clusters[i].centroid[j][0], clusters[i].centroid[j][1], clusters[i].centroid[j][2]])
        centroid_trk.append(centroid)
    streamlines = centroid_trk
    G_pres = nx.Graph() #G_(k)
    #clusters
    clusters_prev = []
    clusters_pres = [] 
    stream_list = []
    # print(len(streamlines), stream_list)
    #When to stop? 1.All points are processed, in other words all (len(), True/False)
    #Will it reach a point where all entries are (-,False) ->solution increase any random
    assign_cluster = []
    for stream_i in range(len(streamlines)):
        stream_list.append([0,True]) #True signals if the index is going to be increased
        a_c = []
        for s_i in range(len(streamlines[stream_i])):
            a_c.append(-1)
        assign_cluster.append(a_c)

    dic_T = {} #to store all events at each points

    for i in range(len(streamlines)):
        dic_T[i] = {0: [Event("appear")], len(streamlines[i])-1 : [Event("disappear")]}

    for i in range(len(streamlines)):    
        for j in range(i+1,len(streamlines)):
            dic_t1, dic_t2 = findConnectDisconnectEvents(i, j, streamlines[i], streamlines[j], eps)        
            for key in dic_t1.keys():
                if dic_T[i].get(key):
                    for e in dic_t1[key]:
                        dic_T[i][key].append(e)
                else:
                    dic_T[i][key] = dic_t1[key]


            for key in dic_t2.keys():
                if dic_T[j].get(key):
                    for e in dic_t2[key]:
                        dic_T[j][key].append(e)
                else:
                    dic_T[j][key] = dic_t2[key]

    #process each events updating dynamic graphs
    process_flag = True
    cluster_id = -1
    del_nodes = []
    while(process_flag):
        process_flag = False
        #find not eligible trajectories to be processed and incremented
        for stream_i in range(len(streamlines)):
            if stream_list[stream_i][0] >= len(streamlines[stream_i]):
                stream_list[stream_i][1] = False
            if dic_T[stream_i].get(stream_list[stream_i][0]):
                events = dic_T[stream_i][stream_list[stream_i][0]]
                for e in events:
                    if e.event == "connect" and stream_list[e.trajectory][0] < e.t:
                        stream_list[stream_i][1] = False
    #                     print("connect", stream_i,stream_list[e.trajectory][0], e.t)
                        break
                    elif e.event == "disconnect" and stream_list[e.trajectory][0] < e.t:
                        stream_list[stream_i][1] = False
    #                     print("disconnect", stream_i,stream_list[e.trajectory][0], e.t)
                        break
        #process all eligible trajectories
    #     print("test", stream_list)
        all_false_count = 0
        for stream_i in range(len(streamlines)):
            if not stream_list[stream_i][1]:
                all_false_count += 1
        if all_false_count == len(streamlines):        
            for stream_i in range(len(streamlines)):
                if  stream_list[stream_i][0] < len(streamlines[stream_i]):
                    stream_list[stream_i][1] = True
                    break
        for stream_i in range(len(streamlines)):
            if stream_list[stream_i][1]:
                process_flag = True
                if dic_T[stream_i].get(stream_list[stream_i][0]):
                    events = dic_T[stream_i][stream_list[stream_i][0]]
                    for e in events:
#                         print(e.event, stream_i, e.trajectory)
                        if e.event == "appear":
                            G_pres.add_node(stream_i)
                        elif e.event == "connect":
                            G_pres.add_node(stream_i)
                            G_pres.add_node(e.trajectory)
                            G_pres.add_edge(stream_i, e.trajectory)
                        elif e.event == "disconnect":
                            try:
                                G_pres.remove_edge(stream_i, e.trajectory)
                            except:
                                pass
                        elif e.event == "disappear":
                            del_nodes.append (stream_i)

        #connected component 
        clusters_pres = list(nx.connected_components(G_pres))
        for cluster_pres in clusters_pres:
            if cluster_pres not in clusters_prev:   

                cluster_id += 1
                for cluster_traj in cluster_pres:
                    if stream_list[cluster_traj][0] < len(streamlines[cluster_traj]):
                        assign_cluster[cluster_traj][stream_list[cluster_traj][0]] = cluster_id

            else:
                for cluster_traj in cluster_pres:
                    if stream_list[cluster_traj][0] < len(streamlines[cluster_traj]) and assign_cluster[cluster_traj][stream_list[cluster_traj][0]] == -1:
                        assign_cluster[cluster_traj][stream_list[cluster_traj][0]] = assign_cluster[cluster_traj][stream_list[cluster_traj][0] -1]


        #prepare for next iteration    
        for stream_i in range(len(streamlines)):
            if stream_list[stream_i][1]:
                stream_list[stream_i][0] = stream_list[stream_i][0] + 1

        for d_node in del_nodes:
            if nx.is_isolate(G_pres, d_node):
                G_pres.remove_node(d_node)
                del_nodes.remove(d_node)
        clusters_prev = []
        for cluster_pres in clusters_pres:
            clusters_prev.append(cluster_pres)


        for stream_i in range(len(streamlines)):
            stream_list[stream_i][1] = True

    count_trajectories = {}
    delete_cluster = set([])
    for stream_i in range(len(streamlines)):
        unique_cluster = list(dict.fromkeys(assign_cluster[stream_i]))
        for uc in unique_cluster: 
            if count_trajectories.get(uc):
                count_trajectories[uc] += cluster_map[stream_i]
            else:
                count_trajectories[uc] = cluster_map[stream_i]
    for (x,y) in count_trajectories.items():
        if y <= delta :
            delete_cluster.add(x)            
    for stream_i in range(len(streamlines)):
        for s_i in range(len(streamlines[stream_i])):
    #         print(assign_cluster[stream_i][s_i])
            if assign_cluster[stream_i][s_i] in delete_cluster:
    #             print("True")
                assign_cluster[stream_i][s_i] = -2
    del_s_id = []
    for stream_i in range(len(streamlines)):
        if all(i == -2 for i in assign_cluster[stream_i]):  
            del_s_id.append(stream_i)
        else:
            for s_i in range(len(streamlines[stream_i])):
                if s_i != 0 and assign_cluster[stream_i][s_i] == -2:
                    assign_cluster[stream_i][s_i] = assign_cluster[stream_i][s_i-1]
            for s_i in range(len(streamlines[stream_i])-1, -1, -1):
                if assign_cluster[stream_i][s_i] == -2:
                    assign_cluster[stream_i][s_i] = assign_cluster[stream_i][s_i+1]

    #Reeb Graph construction from bundles
    R = nx.Graph()
    G_nodes = nx.Graph()
    cluster_edge = {}
    node_loc = {} #with location
    node_id = 0


    for stream_i in range(len(streamlines)):
        if stream_i not in del_s_id:
            unique_cluster = list(dict.fromkeys(assign_cluster[stream_i]))
            if len(unique_cluster) == 1:

                if not cluster_edge.get(unique_cluster[0]):
                    R.add_edge(node_id, node_id + 1)
                    R[node_id][node_id + 1]['weight'] = count_trajectories[unique_cluster[0]]/sum(cluster_map.values())
                    cluster_edge[unique_cluster[0]] = [node_id, node_id + 1]
                    node_id += 2
            for uc in range(len(unique_cluster)-1):
                if not cluster_edge.get(unique_cluster[uc]):
                    R.add_edge(node_id, node_id + 1)
                    R[node_id][node_id + 1]['weight'] = count_trajectories[unique_cluster[uc]]/sum(cluster_map.values())
                    cluster_edge[unique_cluster[uc]] = [node_id, node_id + 1]            
                    node_id += 2
                if not cluster_edge.get(unique_cluster[uc + 1]):
                    R.add_edge(node_id, node_id + 1)
                    R[node_id][node_id + 1]['weight'] = count_trajectories[unique_cluster[uc + 1]]/sum(cluster_map.values())
                    cluster_edge[unique_cluster[uc + 1]] = [node_id, node_id + 1]            
                    node_id += 2           
    
#     print(R.edges.data())
#     print(assign_cluster)

    #node location
    for stream_i in range(len(streamlines)):
        if len(assign_cluster[stream_i]) != 0 and stream_i not in del_s_id:
            x = assign_cluster[stream_i][len(streamlines[stream_i]) - 1]
            if cluster_edge[x][1] in node_loc.keys():
                node_loc[cluster_edge[x][1]].append(streamlines[stream_i][len(streamlines[stream_i]) - 1])
            else:                
                node_loc[cluster_edge[x][1]] = [streamlines[stream_i][len(streamlines[stream_i]) - 1]]
            y = assign_cluster[stream_i][0]
            if cluster_edge[y][0] in node_loc.keys():
                node_loc[cluster_edge[y][0]].append(streamlines[stream_i][0])
            else:                
                node_loc[cluster_edge[y][0]] = [streamlines[stream_i][0]]

            begin = y

            for s_i in range(1, len(streamlines[stream_i])):

                if assign_cluster[stream_i][s_i] != begin:
                    if cluster_edge[begin][1] in node_loc.keys():
                        node_loc[cluster_edge[begin][1]].append(streamlines[stream_i][s_i - 1])
                    else:
                        node_loc[cluster_edge[begin][1]] = [streamlines[stream_i][s_i - 1]]
                    begin = assign_cluster[stream_i][s_i]
                    if cluster_edge[begin][0] in node_loc.keys():
                        node_loc[cluster_edge[begin][0]].append(streamlines[stream_i][s_i])
                    else:                                             
                        node_loc[cluster_edge[begin][0]] = [streamlines[stream_i][s_i]]             

    node_loc_final = {}
    for node_key in node_loc.keys():
#         print(node_loc[node_key])
        n_x = 0
        n_y = 0
        n_z = 0
        for nk in node_loc[node_key]:
            n_x += nk[0]
            n_y += nk[1]
            n_z += nk[2]
        node_loc_final[node_key] = [n_x / len(node_loc[node_key]),n_y / len(node_loc[node_key]),n_z / len(node_loc[node_key])]
#         node_loc_final[node_key] = node_loc[node_key][0]
        

    for stream_i in range(len(streamlines)):
        if stream_i not in del_s_id:
            unique_cluster = list(dict.fromkeys(assign_cluster[stream_i]))
            for uc in range(len(unique_cluster)-1):

                dist1 = distance ( node_loc_final[cluster_edge[unique_cluster[uc]][1]], node_loc_final [cluster_edge[unique_cluster[uc + 1]][0]])
                dist2 = distance(node_loc_final[cluster_edge[unique_cluster[uc]][1]], node_loc_final [cluster_edge[unique_cluster[uc + 1]][1]])
                if dist1 < dist2:
                    G_nodes.add_edge( cluster_edge[unique_cluster[uc]][1], cluster_edge[unique_cluster[uc + 1]][0])
                else:
                    G_nodes.add_edge( cluster_edge[unique_cluster[uc]][1], cluster_edge[unique_cluster[uc + 1]][1])


    merged_nodes = list(nx.connected_components(G_nodes))
    node_map = {}

    for cluster in merged_nodes:

        if len(cluster)>1:        
            for c in cluster:
                node_map[c] = node_id
                if node_id in node_loc_final :
                    node_loc_final[node_id] = [node_loc_final[node_id][0]/2 + node_loc_final[c][0]/2, node_loc_final[node_id][1]/2 + node_loc_final[c][1]/2, node_loc_final[node_id][2]/2 + node_loc_final[c][2]/2]
                else:
                    node_loc_final[node_id] = node_loc_final[c]
                    
            node_id += 1
    H = nx.relabel_nodes(R, node_map)
#     print(node_map, H.edges.data())
    G_nodes = nx.Graph()
    count_del_edge = 0
    # threshold on length of edge (alpha) and edge weight (delta)
    for (n1, n2) in list(H.edges):
        if (distance(node_loc_final[n1], node_loc_final[n2])) < alpha:
            G_nodes.add_edge(n1,n2)            
            count_del_edge += 1
#     print("here",count_del_edge)
    merged_nodes = list(nx.connected_components(G_nodes))
    node_map = {}

    for cluster in merged_nodes:

        if len(cluster)>1:        
            for c in cluster:
                node_map[c] = node_id
                if not node_id in node_loc_final:
                    node_loc_final[node_id] = node_loc_final[c]
                else:
                    node_loc_final[node_id] = [node_loc_final[node_id][0]/2 + node_loc_final[c][0]/2, node_loc_final[node_id][1]/2 + node_loc_final[c][1]/2, node_loc_final[node_id][2]/2 + node_loc_final[c][2]/2]
            node_id += 1
    R = nx.relabel_nodes(H, node_map)
#         if (edge_weight()< delta):
#             H.remove_edge(n1, n2)
#     R.remove_nodes_from(list(nx.isolates(R)))
#     print("assign_cluster", assign_cluster)
#     print("cluster_edge",cluster_edge)
#     print("count_trajectories",count_trajectories)
    R.remove_edges_from(nx.selfloop_edges(R))
    R.remove_nodes_from(list(nx.isolates(R)))
    return R, node_loc_final
        
