import numpy as np
from muon_utils import *
from datetime import datetime
import itertools, gc, os, h5py, shutil


# Simulation parameters
global_start = datetime.now()
# visualise = True  # whether to plot detector histograms and 3d voxel models
save_out = True  # whether to save detectors and voxel models in files
input_dir = 'data/in'
thresh = 0.4  # visualisation threshold after simulation
size = 1  # voxel size
delta_alpha = 1.0  # .0001  # parameter to represent the empty voxels
shift = 0  # np.array([20, 20, 0])

det_nx = 3  # number of detectors along axis (NxN)
multi_detect = det_nx - 1  # number of intersecting detectors in the anomalous voxel
scene_shape = (32, 32, 32)
scale = scene_shape[2] / 20

angle_bin = 3 * scene_shape[2]  # binning of the angle along axis
# min angle bin number should be 2*np.sqrt(2)*h -- furthest voxel angular size is 1 bin
angle_params = (-1, 1, angle_bin, -1, 1, angle_bin)  # angular binning for tan_x and tan_y
dx = scene_shape[0] / det_nx / 2
location_params = (dx, scene_shape[0] - dx, det_nx, dx, scene_shape[0] - dx, det_nx, -dx)  # x&y locations and number of detectors

# # defining the voxels
start = datetime.now()
print('Simulation of {}x{}x{} voxel model with {}x{} grid of detectors, minimum {} of intersecting detectors \n'.format(*scene_shape,det_nx,det_nx,multi_detect))
volo_init, volo_true = create_default_voxel_setting(scene_shape, delta_alpha, shift, size=size)
list_init = voxel_array_to_list(volo_init, size=size)
list_true = voxel_array_to_list(volo_true, size=size)
detectors, voxel_crosses_list = create_detectors(angle_params, location_params, list_true, list_init, size=size,
                                                 detector_shift=shift)
print('initialization time:', datetime.now()-start)
volo_pred = None
detectors_inout = None
# Saving detector initial states
if save_out:
    if not os.path.exists(input_dir): os.mkdir(input_dir)
    if os.path.exists(input_dir + '/3d_output'): shutil.rmtree(input_dir + '/3d_output')
    os.mkdir(input_dir + '/3d_output')
    save_voxel_model(volo_init, format='.dat', file_name=input_dir + '/3d_output/Model_initial')
    save_voxel_model(volo_true, format='.dat', file_name=input_dir + '/3d_output/Model_true')
    if os.path.exists(input_dir + '/detectors'): shutil.rmtree(input_dir + '/detectors')
    os.mkdir(input_dir + '/detectors')
    for i_d, det in detectors.items():
        save_detector(input_dir + '/detectors/det_init.txt', det, i_d)
