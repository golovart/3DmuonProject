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

outf.write('Simulation with {}x{} grid of detectors, minimum {} of intersecting detectors \n\n'.format(det_nx,det_nx,multi_detect))
for s in [16, 20, 32, 40, 64]:
    scene_shape = (s, s, s, 2)
    scale = scene_shape[2] / 20
    angle_bin = 3 * scene_shape[2]  # binning of the angle along axis
    # min angle bin number should be 2*np.sqrt(2)*h -- furthest voxel angular size is 1 bin
    angle_params = (-1, 1, angle_bin, -1, 1, angle_bin)  # angular binning for tan_x and tan_y
    dx = scene_shape[0] / det_nx / 2
    location_params = (dx, scene_shape[0] - dx, det_nx, dx, scene_shape[0] - dx, det_nx, -dx)  # x&y locations and number of detectors
    # location_params = (3.5,16.5,det_nx, 3.5,16.5,det_nx, -3)

    # # defining the voxels
    start = datetime.now()
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
    num_steps = 50
    lr = 0.003 / det_nx / scale
    list_pred = np.copy(list_init)
    for step in range(num_steps):
        detectors, list_pred = grad_step(step, lr, detectors, list_pred, voxel_crosses_list, multi_det=multi_detect,
                                         loss_function='l2', n_decay=10, verbose=False)
    volo_pred = voxel_list_to_array(list_pred, size=size)
    outf.write('iteration time for {}x{}x{} voxels: '.format(s, s, s)+str(datetime.now() - start)+'\n')
    outf.write('Remaining voxel error with {} threshold: {:.2f}'.format(thresh,
                                                                   rve_score(volo_true, volo_init, volo_pred > thresh))+'\n')
    thr_range = np.arange(0.25, 0.76, 0.05)
    rve_range = [rve_score(volo_true, volo_init, volo_pred > th) for th in thr_range]
    i_best = np.argmin(rve_range)
    outf.write('For {}x{}x{} voxels best score:\n\t{:.2f} with threshold {:.2f}'.format(s, s, s, rve_range[i_best],
                                                                                        thr_range[i_best])+'\n')
    outf.write('\n')

    del volo_init, volo_pred, volo_true, list_pred, list_init, list_true, detectors, voxel_crosses_list
    gc.collect()
    # if s>20: break
outf.write('full running time: '+str(datetime.now()-global_start)+'\n')
outf.close()
