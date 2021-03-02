import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from muon_utils import *
import os, shutil

# Simulation parameters
global_start = datetime.now()
# plt.ion()

input_dir, output_dir, det_nx, det_ny, scene_shape, num_steps, lr, thresh, reiterate, det_id, visualise = process_arguments(sys.argv)

size = 1  # voxel size
delta_alpha = 1.0  # parameter to represent the empty voxels
shift = 0  # np.array([20, 20, 0])

detectors = None
if input_dir['read'] or input_dir['save'] or reiterate:
    if not input_dir['read']:
        multi_detect = det_nx - 1  # number of intersecting detectors in the anomalous voxel
        angle_bin = 3 * scene_shape[2]  # binning of the angle along axis
        # min angle bin number should be 2*np.sqrt(2)*h -- furthest voxel angular size is 1 bin
        angle_params = (-1, 1, angle_bin, -1, 1, angle_bin)  # angular binning for tan_x and tan_y
        dx = scene_shape[0] / det_nx / 2
        location_params = (dx, scene_shape[0] - dx, det_nx, dx, scene_shape[0] - dx, det_nx, -dx)  # x&y locations and number of detectors
        # # defining the voxels
        start = datetime.now()
        print('Simulation of {}x{}x{} voxel model with {}x{} grid of detectors, minimum {} of intersecting detectors \n'.format(
            *scene_shape, det_nx, det_nx, multi_detect))
        volo_init, volo_true = create_default_voxel_setting(scene_shape, delta_alpha, shift, size=size)
        list_init = voxel_array_to_list(volo_init, size=size)
        list_true = voxel_array_to_list(volo_true, size=size)
        detectors, voxel_crosses_list = create_detectors(angle_params, location_params, list_true, list_init, size=size,
                                                         detector_shift=shift)
        print('initialization time:', datetime.now() - start)
    else:
        detectors = read_detectors(input_dir['path'] + '/detectors', detector_state='init')
        model_name = [m for m in os.listdir(input_dir['path'] + '/3d_output') if 'init' in m][0]
        volo_init = read_voxel_model(input_dir['path'] + '/3d_output/' + model_name)
        model_name = [m for m in os.listdir(input_dir['path'] + '/3d_output') if 'true' in m][0]
        volo_true = read_voxel_model(input_dir['path'] + '/3d_output/' + model_name)
        scene_shape = volo_true.shape
        det_nx = int(np.sqrt(max(detectors.keys())))
        scale = scene_shape[2] / 32
        multi_detect = det_nx - 1
        list_init = voxel_array_to_list(volo_init, size=size)
        list_true = voxel_array_to_list(volo_true, size=size)
        if reiterate: detectors, voxel_crosses_list = voxels_vs_detectors(list_true, detectors, size=size)

    # visualising initial detector
    if visualise['det']:
        visualise_detector(detectors[det_id], i_det=det_id, iterated=False)
    if visualise['vox']:
        delta_vis = 0.8  # color of unknown structures is different for better visibility
        volo_true_vis = np.copy(volo_init)
        volo_true_vis += delta_vis * np.where(np.abs(volo_true - volo_init), delta_vis, 0)
        visualise_voxels(volo_true_vis, inv=False, figsize=(5, 5))

    if input_dir['save']:
        if not os.path.exists(input_dir['path']): os.mkdir(input_dir['path'])
        if os.path.exists(input_dir['path'] + '/3d_output'): shutil.rmtree(input_dir['path'] + '/3d_output')
        os.mkdir(input_dir['path'] + '/3d_output')
        save_voxel_model(volo_init, format='.dat', file_name=input_dir['path'] + '/3d_output/Model_initial')
        save_voxel_model(volo_true, format='.dat', file_name=input_dir['path'] + '/3d_output/Model_true')
        if os.path.exists(input_dir['path'] + '/detectors'): shutil.rmtree(input_dir['path'] + '/detectors')
        os.mkdir(input_dir['path'] + '/detectors')
        for i_d, det in detectors.items():
            save_detector(input_dir['path'] + '/detectors/det_init.txt', det, i_d)

    if reiterate:
        start = datetime.now()
        list_pred = np.copy(list_init)
        for step in range(num_steps):
            detectors, list_pred = grad_step(step, lr, detectors, list_pred, voxel_crosses_list, multi_det=multi_detect,
                                             loss_function='l2', n_decay=10, verbose=False)
        volo_pred = voxel_list_to_array(list_pred, size=size)
        print('iteration time:', datetime.now() - start)

if output_dir['read'] and not reiterate:
    if visualise['det']:
        assert detectors is not None
        detectors_tmp = read_detectors(output_dir['path'] + '/detectors', detector_state='final')
        assert detectors.keys() == detectors_tmp.keys()
        for i_d in detectors.keys():
            detectors[i_d]['dir_mass_pred'] = detectors_tmp[i_d]['dir_mass_pred']
    model_name = [m for m in os.listdir(output_dir['path'] + '/3d_output') if 'sim' in m][0]
    volo_pred = read_voxel_model(output_dir['path'] + '/3d_output/' + model_name)
    list_pred = voxel_array_to_list(volo_pred, size=size)

# # Saving the results
if output_dir['save'] or (reiterate and output_dir['path'] is not None):
    if not os.path.exists(output_dir['path']): os.mkdir(output_dir['path'])
    if os.path.exists(output_dir['path'] + '/3d_output'): shutil.rmtree(output_dir['path'] + '/3d_output')
    os.mkdir(output_dir['path'] + '/3d_output')
    save_voxel_model(volo_pred, format='.dat', file_name=output_dir['path']+'/3d_output/Model_simulated')
    if os.path.exists(output_dir['path'] + '/detectors'): shutil.rmtree(output_dir['path'] + '/detectors')
    os.mkdir(output_dir['path'] + '/detectors')
    for i_d, det in detectors.items():
        save_detector(output_dir['path']+'/detectors/det_final.txt', det, i_d, sim_types=['pred'])

# Visualising final states
if visualise['det']:
    visualise_detector(detectors[det_id], i_det=det_id, iterated=True)
if visualise['vox']:
    det_coords = None if not visualise['det'] else np.array([d['coord'] for d in detectors.values()])
    visualise_voxels(volo_pred, inv=False, figsize=(5,5), det_coords=det_coords)
    vox_pred_thresh = np.where(volo_pred > thresh, delta_alpha, 1.0 - delta_alpha)
    visualise_voxels(vox_pred_thresh, inv=False, figsize=(5, 5), det_coords=det_coords)
    if input_dir['read'] or input_dir['save'] or reiterate:
        visualise_voxels((np.abs(volo_true - vox_pred_thresh)).clip(0, 1), inv=False, figsize=(5, 5))
        print('Initial difference (in voxels): {:.2f}'.format(np.abs(volo_true-volo_init).sum()))
        print('Remaining voxel error with {} threshold: {:.2f}'.format(thresh,
                                                                       rve_score(volo_true, volo_init, volo_pred > thresh)))
print('full running time: '+str(datetime.now()-global_start))
plt.show()
