import nibabel as nib
import os
import robustReebConstruction_avg_loc_qb as rc
import networkx as nx
import pickle
#path name for CC files
trk_file_path = "/media/hdd2/shailja/Crash/CC_ref/"
trk_files = os.listdir(trk_file_path)
for file in trk_files:
    if ".trk.gz" in file and ".txt" not in file:
        print("Processing ...",file)
        cc_streamlines =  nib.streamlines.load(trk_file_path+file)
        print("n = ",len(cc_streamlines.streamlines))        
        eps =2.5
        delta = 3
        tau = 5
        H, node_loc,cluster_edge = rc.constructRobustReeb(cc_streamlines.streamlines, eps, delta, tau)
        #write H
        nx.write_gpickle(H, trk_file_path+"H_CC_ref_"+file.split("_")[0]+"_"+file.split("_")[1]+".gpickle")
        #save node_loc dict
        with open(trk_file_path+"node_loc_CC_ref_"+file.split("_")[0]+"_"+file.split("_")[1]+".pickle", 'wb') as handle:
            pickle.dump(node_loc, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
