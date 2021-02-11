import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import *
import h5py, os, shutil

# Double voxel merging
# Simulation parameters
size = 1  # voxel size
fact = 1  # enlarging the voxel size (and angular bins)
size *= fact
delta_alpha = 0  # .0001  # small parameter to represent the empty voxels
scene_shape = (8,8,8,2)
scale = scene_shape[2]/20
angle_bin, det_nx = 3*scene_shape[2], 3  # binning of the angle along axis and number of detectors along axis (NxN)
# min angle bin number should be 2*np.sqrt(2)*h -- furthest voxel angular size is 1 bin
angle_params = (-1,1,angle_bin//fact, -1,1,angle_bin//fact)  # angular binning for tan_x and tan_y
location_params = (3*scale,17*scale,det_nx, 3*scale,17*scale,det_nx, -3)  # x&y locations and number of detectors
# location_params = (3.5,16.5,det_nx, 3.5,16.5,det_nx, -3)
multi_detect = det_nx-1  # number of intersecting detectors in the anomalous voxel
thresh = 0.6  # visualisation threshold after simulation
shift = 0 # np.array([20, 20, 0])


def line_rule(x,y,z, p1=None, p2=None, delta=1.0, R=20):
    p = np.array([x,y,z])*delta
    l = p-p1; l12 = p2-p1
    d1 = -np.dot(p2-p1,p1); d2 = -np.dot(p2-p1, p2)
    if np.linalg.norm(l)<delta: return True
    return (np.linalg.norm(np.cross(l,l12))/np.linalg.norm(l12)<R and np.dot(p2-p1,p)+d2<0 and np.dot(p2-p1,p)+d1>0) or np.linalg.norm(p-p1)<R or np.linalg.norm(p-p2)<R
# (np.linalg.norm(np.cross(l,l12))/np.linalg.norm(l12)<R and np.dot(p2-p1,p)+d2<0 and np.dot(p2-p1,p)+d1>0)


# # defining the voxels
start = datetime.now()
volo_init = np.ones(scene_shape)
volo_init[...,0] -= delta_alpha
s_t, r_t = np.array((4,4,4)), 2
volo_init = add_empty_sphere(volo_init, s_t, r_t, (0,0))
save_voxel_model(1-volo_init[...,0], format='.dat', file_name='data/test_3dSphere')
# volo_read = read_voxel_model('data/3dModel_256x256x256.dat')
# Spheres
# print('save time:', datetime.now()-start)
s1, s2, s3, s4, s5 = np.array((3.5,3.5,5.5)), np.array((14.5,9.5,15.5)), np.array((12.5,2.5,2.5)), np.array((9.5,15.5,3.5)), np.array((9,6.5,10.5))
if shift is not None:
    s1 += shift; s2 += shift; s3 += shift; s4 += shift; s5 += shift
s1 *= scale; s2 *= scale; s3 *= scale; s4 *= scale; s5 *= scale
r1, r2, r3, r4, rl = 1.5, 2.5, 2, 2, 1
r1 *= scale; r2 *= scale; r3 *= scale; r4 *= scale; rl *= scale
volo_init = add_empty_sphere(volo_init, s1, r1, (delta_alpha,0))
# save_voxel_model(voxel_array_to_list(volo_init, size=size), size=size, name='data/3dModel.txt', delta=delta_alpha)
volo_init = add_empty_tunnel(volo_init, s1, s2, rl, (delta_alpha,0))
volo_init = add_empty_sphere(volo_init, s2, r2, (delta_alpha,0))
volo_init = add_empty_tunnel(volo_init, s2, s3, rl, (delta_alpha,0))
volo_init = add_empty_sphere(volo_init, s3, r3, (delta_alpha,0))
volo_init = add_empty_tunnel(volo_init, s3, s5, rl, (delta_alpha,0))
volo_init = add_empty_tunnel(volo_init, s2, s4, rl, (delta_alpha,0))
volo_init = add_empty_sphere(volo_init, s4, r4, (delta_alpha,0))

volo_true = np.copy(volo_init)
volo_true = add_empty_sphere(volo_true, (np.array((15.5,13.5,3.5))+shift)*scale, 2.5*scale, (delta_alpha,1))
volo_true = add_empty_sphere(volo_true, (np.array((12,5,15.5))+shift)*scale, 1.8*scale, (delta_alpha,1))
volo_true = add_empty_sphere(volo_true, (np.array((4,8,5.5))+shift)*scale, 2*scale, (delta_alpha,1))



# def merge(vox_array, factor=2, delta=0.01):
#     big_shape = np.ceil(np.array(vox_array.shape)/factor).astype(int)
#     vox_big = np.zeros(big_shape)
#     for i in range(big_shape[0]):
#         for j in range(big_shape[1]):
#             for k in range(big_shape[2]):
#                 vox_big[i,j,k] = np.sum(vox_array[i*factor:(i+1)*factor, j*factor:(j+1)*factor, k*factor:(k+1)*factor])/factor**3
#     # vox_big = np.where(vox_big>0.5, 1-delta, delta)
#     # TODO: implement threshold before visualisation
#     return vox_big
#
#
# # simulation with big voxels
# volo_true = merge(volo_true, factor=fact, delta=delta_alpha) if fact > 1 else np.copy(volo_true) # /fact**3
# volo_init = merge(volo_init, factor=fact, delta=delta_alpha) if fact > 1 else np.copy(volo_init)  # /fact**3

# visualise_voxels(volo_true[...,0], inv=True, figsize=(5,5))

list_init = voxel_array_to_list(volo_init, size=size)
list_true = voxel_array_to_list(volo_true, size=size)
detectors, voxel_crosses_list = create_detectors(angle_params, location_params, list_true, list_init, size=size, detector_shift=shift)
print('initialization time:', datetime.now()-start)

# for i_d, det in enumerate(detectors):
i_d = 1; det = detectors[i_d]
visualise_detector(det, angle_params, i_det=i_d, iterated=False)
# Saving detector initial states
# if os.path.exists('data/detectors'): shutil.rmtree('data/detectors')
# os.mkdir('data/detectors')
# for i_d, det in enumerate(detectors):
#     save_detector('data/detectors/det_init.txt', det, angle_params, i_d)

start = datetime.now()
num_steps = 50
lr = 0.003*fact/det_nx/scale#*2
list_pred = np.copy(list_init)
for s in range(num_steps):
    detectors, list_pred = grad_step(s, lr, detectors, list_pred, voxel_crosses_list, multi_det=multi_detect, loss_function='l2', n_decay=10, verbose=False)

print('iteration time:', datetime.now()-start)
visualise_detector(detectors[i_d], angle_params, i_det=i_d, iterated=True)
# to visualise unknown structures in "true" scheme
delta_vis = (0.2,1)
volo_true = add_empty_sphere(volo_true, (np.array((15.5, 13.5, 3.5)) + shift) * scale, 2.5 * scale, delta_vis)
volo_true = add_empty_sphere(volo_true, (np.array((12, 5, 15.5)) + shift) * scale, 1.8 * scale, delta_vis)
volo_true = add_empty_sphere(volo_true, (np.array((4, 8, 5.5)) + shift) * scale, 2 * scale, delta_vis)
visualise_voxels(volo_true[...,0], inv=True, figsize=(5, 5))
visualise_voxels(voxel_list_to_array(list_pred, size=size)[...,0].clip(0,1), inv=True, figsize=(5,5), det_coords=np.array([d['coord'] for d in detectors])/fact)

vox_pred_thresh = np.where(voxel_list_to_array(list_pred, size=size).clip(0,1)>thresh, 1.0-delta_alpha, delta_alpha)
volo_true[...,0] = np.where(volo_true[...,0] > thresh, 1.0 - delta_alpha, delta_alpha)
visualise_voxels(vox_pred_thresh[...,0], inv=True, figsize=(5,5), det_coords=np.array([d['coord'] for d in detectors])/fact)
visualise_voxels((np.abs(volo_true - vox_pred_thresh))[...,0].clip(0, 1), inv=False, figsize=(5, 5))
print('iteration+visualisation time:', datetime.now()-start, end='\n\n')
print('Initial difference (in voxels): {:.2f}'.format(np.abs(volo_true-volo_init)[...,0].sum()))
print('Error size (in wrong voxels) for {}x{} detectors: {:.2f}'.format(det_nx, det_nx, np.abs(volo_true - vox_pred_thresh)[...,0].sum()))

# Saving the results
# if os.path.exists('data/3d_output'): shutil.rmtree('data/3d_output')
# os.mkdir('data/3d_output')
# save_voxel_model(1-voxel_list_to_array(list_pred, size=size)[...,0].clip(0,1), format='.dat', file_name='data/3d_output/Model_simulated')
# for i_d, det in enumerate(detectors):
#     save_detector('data/detectors/det_final.txt', det, angle_params, i_d, sim_types=['pred'])

# for i_d in range(len(detectors)):
#     visualise_detector(detectors[i_d], angle_params, i_det=i_d, iterated=True)