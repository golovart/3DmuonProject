import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt
# from matplotlib import cm, colors
# import itertools, gc, os
from datetime import datetime
from utils import *

# Simulation parameters
size = 1  # voxel size
delta_alpha = 0.001  # small parameter to represent the empty voxels
angle_params = (-1,1,60, -1,1,60)  # angular binning for tan_x and tan_y
location_params = (4.5,15.5,3, 5.5,15.5,3, -3)  # x&y locations and number of detectors


# # defining the voxels
start = datetime.now()
volo_init = np.ones((20, 20, 20)) - delta_alpha
for i in range(volo_init.shape[0]):
    for j in range(volo_init.shape[1]):
        for k in range(volo_init.shape[2]):
            if (i - 7.5) ** 2 + (j - 7.5) ** 2 + (k - 7.5) ** 2 < 30: volo_init[i, j, k] = delta_alpha

volo_true = np.copy(volo_init)
for i in range(volo_init.shape[0]):
    for j in range(volo_init.shape[1]):
        for k in range(volo_init.shape[2]):
            if (i - 4.5) ** 2 + (j - 2.5) ** 2 + (k - 3.5) ** 2 < 2.5: volo_true[i, j, k] = delta_alpha
            if (i - 15.5) ** 2 + (j - 10.5) ** 2 + (k - 15.5) ** 2 < 3.5: volo_true[i, j, k] = delta_alpha

# visualise_voxels(volo_true, inv=True, figsize=(7, 7))

# # defining the detectors
list_init = voxel_array_to_list(volo_init, size=size)
list_true = voxel_array_to_list(volo_true, size=size)

# test_vec = line_x_all_cubes(list_init[:,:-1], (5,5,5), (-0.66667,-0.66667,0.33333), r=0.5)
# # detectors = {}
# test_loop = voxels_per_direction(np.array((5,5,5)), np.array((-0.66667,-0.66667,0.33333)), list_init, size=1)

# location_params = [[2,2,-3], [2,17,-3], [17,2,-3], [8,8,-3]]
detectors, voxel_crosses_list = create_detectors(angle_params, location_params, list_true, list_init, size=size)
# voxel_crosses_list, detectors = cross_voxels_detectors(detectors, list_init, size=1)
# detectors = fill_detectors(detectors, list_true, list_init)
# det_per_vox_list = directions_per_voxel(detectors, list_init, size=1)
print('initialization time:', datetime.now()-start)

# for i_d, det in enumerate(detectors):
i_d = 1; det = detectors[i_d]
visualise_detector(det, angle_params, i_det=i_d, iterated=False)

start = datetime.now()
num_steps = 25
lr = 0.005
list_pred = np.copy(list_init)
for s in range(num_steps):
    detectors, list_pred = grad_step(s, lr, detectors, list_pred, voxel_crosses_list, multi_det=4, loss_function='l2')

print('iteration time:', datetime.now()-start)
visualise_detector(detectors[1], angle_params, i_det=1, iterated=True)
# visualise_voxels(volo_true, inv=True, figsize=(5,5))
# visualise_voxels(voxel_list_to_array(list_pred, size=size).clip(0,1), inv=True, figsize=(5,5), det_coords=np.array([d['coord'] for d in detectors]))
# visualise_voxels(((voxel_list_to_array(list_pred, size=size)>0.5)-delta_alpha).clip(0,1), inv=True, figsize=(5,5))
print('iteration+visualisation time:', datetime.now()-start)
