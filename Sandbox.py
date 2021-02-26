import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from muon_utils import *
# from create_scene import *
import os, shutil

# Simulation parameters
global_start = datetime.now()
read_from_files = True
reiterate = False
visualise = True  # whether to plot detector histograms and 3d voxel models
save_out = False  # whether to save detectors and voxel models in files
thresh = 0.4  # visualisation threshold after simulation
size = 1  # voxel size
delta_alpha = 1.0  # .0001  # parameter to represent the empty voxels
shift = 0  # np.array([20, 20, 0])
input_dir = 'data/in'
output_dir = 'data/out'

if not read_from_files:
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
    detectors_in = None
    detectors_out = None
    # Saving detector initial states
    if save_out:
        if not os.path.exists(input_dir): os.mkdir(input_dir)
        if os.path.exists(input_dir+'/3d_output'): shutil.rmtree(input_dir+'/3d_output')
        os.mkdir(input_dir+'/3d_output')
        save_voxel_model(volo_init, format='.dat', file_name=input_dir+'/3d_output/Model_initial')
        save_voxel_model(volo_true, format='.dat', file_name=input_dir+'/3d_output/Model_true')
        if os.path.exists(input_dir+'/detectors'): shutil.rmtree(input_dir+'/detectors')
        os.mkdir(input_dir+'/detectors')
        for i_d, det in detectors.items():
            save_detector(input_dir+'/detectors/det_init.txt', det, i_d)
else:
    detectors_in = read_detectors(input_dir+'/detectors')
    if os.path.exists(output_dir + '/detectors'): detectors_out = read_detectors(output_dir + '/detectors')
    else: detectors_out = None
    detectors = detectors_in['init']
    volo_init, volo_true, volo_pred = None, None, None
    for model_name in os.listdir(input_dir+'/3d_output'):
        if 'init' in model_name:
            volo_init = read_voxel_model(input_dir+'/3d_output/'+model_name)
        elif 'true' in model_name:
            volo_true = read_voxel_model(input_dir+'/3d_output/'+model_name)
        elif 'sim' in model_name:
            volo_pred = read_voxel_model(input_dir+'/3d_output/'+model_name)
    if os.path.exists(output_dir+'/3d_output'):
        for model_name in os.listdir(output_dir+'/3d_output'):
            if 'sim' in model_name:
                volo_pred = read_voxel_model(output_dir+'/3d_output/'+model_name)
        assert (volo_init is not None) and (volo_true is not None)
    scene_shape = volo_true.shape
    det_nx = int(np.sqrt(max(detectors.keys())))
    scale = scene_shape[2] / 20
    multi_detect = det_nx - 1

    list_init = voxel_array_to_list(volo_init, size=size)
    list_true = voxel_array_to_list(volo_true, size=size)
    detectors, voxel_crosses_list = voxels_vs_detectors(list_true, detectors, size=size)

    if volo_pred is not None: list_pred = voxel_array_to_list(volo_pred, size=size)

# visualising initial detector
if visualise:
    i_d = 1; det = detectors[i_d]
    visualise_detector(det, i_det=i_d, iterated=False)

if volo_pred is None or reiterate:
    start = datetime.now()
    num_steps = 50
    lr = 0.003 / det_nx / scale
    list_pred = np.copy(list_init)
    for step in range(num_steps):
        detectors, list_pred = grad_step(step, lr, detectors, list_pred, voxel_crosses_list, multi_det=multi_detect,
                                         loss_function='l2', n_decay=10, verbose=False)
    volo_pred = voxel_list_to_array(list_pred, size=size)
    print('iteration time:', datetime.now()-start)
else:
    assert detectors_out is not None
    for i_d in detectors.keys():
        detectors[i_d]['dir_mass_pred'] = detectors_out['final'][i_d]['dir_mass_pred']


# # Saving the results
if save_out:
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    if os.path.exists(output_dir + '/3d_output'): shutil.rmtree(output_dir + '/3d_output')
    os.mkdir(output_dir + '/3d_output')
    save_voxel_model(volo_pred, format='.dat', file_name=output_dir+'/3d_output/Model_simulated')
    if os.path.exists(output_dir + '/detectors'): shutil.rmtree(output_dir + '/detectors')
    os.mkdir(output_dir + '/detectors')
    for i_d, det in detectors.items():
        save_detector(output_dir+'/detectors/det_final.txt', det, i_d, sim_types=['pred'])


# Visualising final states
if visualise:
    i_d = 1; visualise_detector(detectors[i_d], i_det=i_d, iterated=True)
    # to visualise unknown structures in "true" scheme
    delta_vis = 0.8
    volo_true = add_empty_sphere(volo_true, (np.array((15.5, 13.5, 3.5)) + shift) * scale, 2.5 * scale, delta_vis)
    volo_true = add_empty_sphere(volo_true, (np.array((12, 5, 15.5)) + shift) * scale, 1.8 * scale, delta_vis)
    volo_true = add_empty_sphere(volo_true, (np.array((4, 8, 5.5)) + shift) * scale, 2 * scale, delta_vis)
    visualise_voxels(volo_true, inv=False, figsize=(5, 5))
    visualise_voxels(volo_pred, inv=False, figsize=(5,5), det_coords=np.array([d['coord'] for d in detectors.values()]))

    vox_pred_thresh = np.where(volo_pred>thresh, delta_alpha, 1.0 - delta_alpha)
    volo_true = np.where(volo_true > thresh, delta_alpha, 1.0 - delta_alpha)
    visualise_voxels(vox_pred_thresh, inv=False, figsize=(5,5), det_coords=np.array([d['coord'] for d in detectors.values()]))
    visualise_voxels((np.abs(volo_true - vox_pred_thresh)).clip(0, 1), inv=False, figsize=(5, 5))
# print('iteration+visualisation time:', datetime.now()-start, end='\n\n')
print('Initial difference (in voxels): {:.2f}'.format(np.abs(volo_true-volo_init).sum()))
print('Remaining voxel error with {} threshold: {:.2f}'.format(thresh,
                                                               rve_score(volo_true, volo_init, volo_pred > thresh)))

print('full running time: '+str(datetime.now()-global_start))
