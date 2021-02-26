from muon_utils import *
from datetime import datetime
import gc

# Simulation parameters
size = 1  # voxel size
delta_alpha = 1.0  # .0001  # parameter to represent the empty voxels
thresh = 0.5  # visualisation threshold after simulation
shift = 0  # np.array([20, 20, 0])
det_nx = 3  # number of detectors along axis (NxN)
multi_detect = det_nx - 1  # number of intersecting detectors in the anomalous voxel
global_start = datetime.now()
outf = open(__file__[:-3]+'_results.txt', 'w')

default_scene = (40,40,40)
outf.write('Simulation of {}x{}x{} voxel model with {}x{} grid of detectors, minimum {} of intersecting detectors \n\n'.format(*default_scene,det_nx,det_nx,multi_detect))
for scene_shape in [default_scene, default_scene+tuple([2])]:
    # scene_shape = (s, s, s, 2)
    scale = scene_shape[2] / 20
    angle_bin = 3 * scene_shape[2]  # binning of the angle along axis
    # min angle bin number should be 2*np.sqrt(2)*h -- furthest voxel angular size is 1 bin
    angle_params = (-1, 1, angle_bin, -1, 1, angle_bin)  # angular binning for tan_x and tan_y
    dx = scene_shape[0] / det_nx / 2
    location_params = (dx, scene_shape[0] - dx, det_nx, dx, scene_shape[0] - dx, det_nx, -dx)  # x&y locations and number of detectors
    # location_params = (3.5,16.5,det_nx, 3.5,16.5,det_nx, -3)

    # # defining the voxels
    start = datetime.now()
    volo_init, volo_true = create_default_voxel_setting(scene_shape, delta_alpha, shift, size=size)
    list_init = voxel_array_to_list(volo_init, size=size)
    list_true = voxel_array_to_list(volo_true, size=size)
    detectors, voxel_crosses_list = create_detectors(angle_params, location_params, list_true, list_init, size=size,
                                                     detector_shift=shift)
    num_steps = 50
    lr = 0.003 / det_nx / scale
    list_pred = np.copy(list_init)
    for step in range(num_steps):
        detectors, list_pred = grad_step(step, lr, detectors, list_pred, voxel_crosses_list, multi_det=multi_detect,
                                         loss_function='l2', n_decay=10, verbose=False)
    volo_pred = voxel_list_to_array(list_pred, size=size)
    if len(scene_shape)<4:  outf.write('No fixed cavities\n')
    else:   outf.write('Fixed known cavities\n')
    outf.write('iteration time: '+str(datetime.now() - start)+'\n')
    outf.write('Remaining voxel error with {} threshold: {:.2f}'.format(thresh,
                                                                   rve_score(volo_true, volo_init, volo_pred > thresh))+'\n')
    thr_range = np.arange(0.25, 0.76, 0.05)
    rve_range = [rve_score(volo_true, volo_init, volo_pred > th) for th in thr_range]
    i_best = np.argmin(rve_range)
    outf.write('Best score:\n\t{:.2f} with threshold {:.2f}'.format(rve_range[i_best],
                                                                                        thr_range[i_best])+'\n')
    outf.write('\n')

    del volo_init, volo_pred, volo_true, list_pred, list_init, list_true, detectors, voxel_crosses_list
    gc.collect()
    # if s>20: break
outf.write('full running time: '+str(datetime.now()-global_start)+'\n')
outf.close()
