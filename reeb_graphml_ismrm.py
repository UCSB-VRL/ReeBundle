import nibabel as nib
import os
import robustReebConstruction as rc
import pickle


trk_file_path = "/media/hdd2/shailja/ismrm_tractogram_comparison/gt_bundles/"
# trk_file_path = "/media/hdd2/shailja/ismrm_tractogram_comparison/test_bundles/segmented_VB/"

trk_files = os.listdir(trk_file_path)
for file in trk_files:
    if ".trk" in file and ".gpickle" not in file:
        print("Processing ...",file)
        bundle_streamlines =  nib.streamlines.load(trk_file_path+file)
        print("n = ",len(bundle_streamlines.streamlines))        
        eps =2.5
        delta = 3
        tau = 5
        
        H, node_loc = rc.constructRobustReeb(bundle_streamlines.streamlines, eps, delta, tau)
        #write H
        with open(trk_file_path+"reeBundles/"+file.split(".")[0]+".gpickle", 'wb') as f:
            pickle.dump(H, f, pickle.HIGHEST_PROTOCOL)
        #save node_loc dict
        with open(trk_file_path+"reeBundles/"+file.split(".")[0]+".pickle", 'wb') as handle:
            pickle.dump(node_loc, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
