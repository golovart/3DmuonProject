import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from muon_utils import *
import os, shutil

# Simulation parameters
visualise = True  # whether to plot detector histograms and 3d voxel models
save_out = True  # whether to save detectors and voxel models in files
size = 1  # voxel size
delta_alpha = 1.0  # .0001  # parameter to represent the empty voxels
thresh = 0.4  # visualisation threshold after simulation
shift = 0  # np.array([20, 20, 0])
det_nx = 3  # number of detectors along axis (NxN)
multi_detect = det_nx - 1  # number of intersecting detectors in the anomalous voxel
scene_shape = (32, 32, 32, 2)
scale = scene_shape[2] / 20
angle_bin = 3 * scene_shape[2]  # binning of the angle along axis
# min angle bin number should be 2*np.sqrt(2)*h -- furthest voxel angular size is 1 bin
angle_params = (-1, 1, angle_bin, -1, 1, angle_bin)  # angular binning for tan_x and tan_y
dx = scene_shape[0] / det_nx / 2
location_params = (dx, scene_shape[0] - dx, det_nx, dx, scene_shape[0] - dx, det_nx, -dx)  # x&y locations and number of detectors
global_start = datetime.now()

# # defining the voxels
start = datetime.now()
print('Simulation of {}x{}x{} voxel model with {}x{} grid of detectors, minimum {} of intersecting detectors \n'.format(*scene_shape[:-1],det_nx,det_nx,multi_detect))
volo_init = np.ones(scene_shape)
volo_init[..., 0] -= delta_alpha
s1, s2, s3, s4, s5 = np.array((3.5, 3.5, 5.5)), np.array((14.5, 9.5, 15.5)), np.array((12.5, 2.5, 2.5)), np.array(
    (9.5, 15.5, 3.5)), np.array((9, 6.5, 10.5))
if shift is not None:
    s1 += shift;        s2 += shift;        s3 += shift;        s4 += shift;        s5 += shift
s1 *= scale;    s2 *= scale;    s3 *= scale;    s4 *= scale;    s5 *= scale
r1, r2, r3, r4, rl = 1.5, 2.5, 2, 2, 1
r1 *= scale;    r2 *= scale;    r3 *= scale;    r4 *= scale;    rl *= scale
volo_init = add_empty_sphere(volo_init, s1, r1, (delta_alpha, 0))
volo_init = add_empty_tunnel(volo_init, s1, s2, rl, (delta_alpha, 0))
volo_init = add_empty_sphere(volo_init, s2, r2, (delta_alpha, 0))
volo_init = add_empty_tunnel(volo_init, s2, s3, rl, (delta_alpha, 0))
volo_init = add_empty_sphere(volo_init, s3, r3, (delta_alpha, 0))
volo_init = add_empty_tunnel(volo_init, s3, s5, rl, (delta_alpha, 0))
volo_init = add_empty_tunnel(volo_init, s2, s4, rl, (delta_alpha, 0))
volo_init = add_empty_sphere(volo_init, s4, r4, (delta_alpha, 0))

volo_true = np.copy(volo_init)
volo_true = add_empty_sphere(volo_true, (np.array((15.5, 13.5, 3.5)) + shift) * scale, 2.5 * scale,
                             (delta_alpha, 1))
volo_true = add_empty_sphere(volo_true, (np.array((12, 5, 15.5)) + shift) * scale, 1.8 * scale, (delta_alpha, 1))
volo_true = add_empty_sphere(volo_true, (np.array((4, 8, 5.5)) + shift) * scale, 2 * scale, (delta_alpha, 1))

list_init = voxel_array_to_list(volo_init, size=size)
list_true = voxel_array_to_list(volo_true, size=size)
detectors, voxel_crosses_list = create_detectors(angle_params, location_params, list_true, list_init, size=size,
                                                 detector_shift=shift)
print('initialization time:', datetime.now()-start)
# saving initial detectors
if visualise:
    i_d = 1; det = detectors[i_d]
    visualise_detector(det, angle_params, i_det=i_d, iterated=False)
# Saving detector initial states
if save_out:
    if os.path.exists('data/3d_output'): shutil.rmtree('data/3d_output')
    os.mkdir('data/3d_output')
    save_voxel_model(volo_init[..., 0], format='.dat', file_name='data/3d_output/Model_initial')
    save_voxel_model(volo_true[..., 0], format='.dat', file_name='data/3d_output/Model_true')
    if os.path.exists('data/detectors'): shutil.rmtree('data/detectors')
    os.mkdir('data/detectors')
    for i_d, det in enumerate(detectors):
        save_detector('data/detectors/det_init.txt', det, angle_params, i_d)

start = datetime.now()
num_steps = 50
lr = 0.003 / det_nx / scale
list_pred = np.copy(list_init)
for step in range(num_steps):
    detectors, list_pred = grad_step(step, lr, detectors, list_pred, voxel_crosses_list, multi_det=multi_detect,
                                     loss_function='l2', n_decay=10, verbose=False)
volo_pred = voxel_list_to_array(list_pred, size=size)
print('iteration time:', datetime.now()-start)

# Visualising final states
if visualise:
    i_d = 1; visualise_detector(detectors[i_d], angle_params, i_det=i_d, iterated=True)
    # to visualise unknown structures in "true" scheme
    delta_vis = (0.8,1)
    volo_true = add_empty_sphere(volo_true, (np.array((15.5, 13.5, 3.5)) + shift) * scale, 2.5 * scale, delta_vis)
    volo_true = add_empty_sphere(volo_true, (np.array((12, 5, 15.5)) + shift) * scale, 1.8 * scale, delta_vis)
    volo_true = add_empty_sphere(volo_true, (np.array((4, 8, 5.5)) + shift) * scale, 2 * scale, delta_vis)
    visualise_voxels(volo_true[...,0], inv=False, figsize=(5, 5))
    visualise_voxels(volo_pred[...,0], inv=False, figsize=(5,5), det_coords=np.array([d['coord'] for d in detectors]))

    vox_pred_thresh = np.where(volo_pred>thresh, delta_alpha, 1.0 - delta_alpha)
    volo_true[...,0] = np.where(volo_true[...,0] > thresh, delta_alpha, 1.0 - delta_alpha)
    visualise_voxels(vox_pred_thresh[...,0], inv=False, figsize=(5,5), det_coords=np.array([d['coord'] for d in detectors]))
    visualise_voxels((np.abs(volo_true - vox_pred_thresh))[...,0].clip(0, 1), inv=False, figsize=(5, 5))
# print('iteration+visualisation time:', datetime.now()-start, end='\n\n')
print('Initial difference (in voxels): {:.2f}'.format(np.abs(volo_true-volo_init)[...,0].sum()))
print('Remaining voxel error with {} threshold: {:.2f}'.format(thresh,
                                                               rve_score(volo_true, volo_init, volo_pred > thresh)))

# # Saving the results
if save_out:
    save_voxel_model(volo_pred[...,0], format='.dat', file_name='data/3d_output/Model_simulated')
    for i_d, det in enumerate(detectors):
        save_detector('data/detectors/det_final.txt', det, angle_params, i_d, sim_types=['pred'])

print('full running time: '+str(datetime.now()-global_start))
