import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from muon_utils import *
import h5py, os, shutil

# Double voxel merging
# Simulation parameters
size = 1  # voxel size
fact = 1  # enlarging the voxel size (and angular bins)
size *= fact
delta_alpha = 0  # .0001  # small parameter to represent the empty voxels
scene_shape = (20,20,20,2)
scale = scene_shape[2]/20
angle_bin, det_nx = 3*scene_shape[2], 3  # binning of the angle along axis and number of detectors along axis (NxN)
# min angle bin number should be 2*np.sqrt(2)*h -- furthest voxel angular size is 1 bin
angle_params = (-1,1,angle_bin//fact, -1,1,angle_bin//fact)  # angular binning for tan_x and tan_y
location_params = (3*scale,17*scale,det_nx, 3*scale,17*scale,det_nx, -3)  # x&y locations and number of detectors
# location_params = (3.5,16.5,det_nx, 3.5,16.5,det_nx, -3)
multi_detect = 1 #det_nx-1  # number of intersecting detectors in the anomalous voxel
thresh = 0.6  # visualisation threshold after simulation
shift = 0 #np.array([20, 20, 0])


# # defining the voxels
start = datetime.now()
volo_earth = np.ones(scene_shape)
volo_init = np.copy(volo_earth)
volo_init[...,0] -= delta_alpha
# s_t, r_t = np.array((4,4,4)), 2
# volo_init = add_empty_sphere(volo_init, s_t, r_t, (0,0))
# save_voxel_model(1-volo_init[...,0], format='.dat', file_name='data/test_3dSphere')
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


# Modifying the detector calculations to remove artifacts
def passed_muons(passed_list: list, voxel_list: np.ndarray, earth_list: np.ndarray = None):
    """
    Calculating amount of 'mass' passed by muons in specific direction (ray)

    :param passed_list: list of voxel ids crossed by this ray
    :param voxel_list: list of voxels (x,y,z,m)
    :return: float
    """
    mass_loc = -2 if voxel_list.shape[1] > 4 else -1
    mass_passed = 0.0
    if earth_list is not None: earth_passed = 0.0
    for i_v in passed_list:
        mass_passed += voxel_list[i_v][mass_loc]
        if earth_list is not None: earth_passed += earth_list[i_v][mass_loc]
    if earth_list is not None: return mass_passed-earth_passed
    return mass_passed


def create_detectors(angle_params, location_params, list_true, list_pred, list_earth=None, size=1, detector_shift=None):
    """
    Creates a set of detectors with defined parameters (angular binning and location coordinates) and finds correspondence between voxels and directions (rays) crossing them for every detector.
    Calculates masses along each direction for true and (initial) predicted 3d models.

    :param angle_params: min, max and n_steps for tan_x and tan_y angular binning
    :param location_params: (x_min, x_max, n_x, y_min, y_max, n_y, z) or tuple (len!=7) of (x,y,z) for each detector
    :param list_true: real 3d model, (Nx*Ny*Nz,4) ndarray of voxel coordinates and masses (x,y,z,m)
    :param list_pred: predicted 3d model, (Nx*Ny*Nz,4) ndarray of voxel coordinates and masses (x,y,z,m)
    :param size: voxel size, default is 1
    :param detector_shift: (x,y,z), not None if all detectors have to be shifted in fixed direction
    :return:
    detectors: list of dictionaries with ('coord','angles','dir_mass_true','dir_mass_pred')
    vox_crosses: list of lists, for each voxel is a list of (detector_id, direction_id) for all the rays that cross corresponding voxel
    """
    mass_loc = -2 if list_pred.shape[1] > 4 else -1
    tx_min, tx_max, n_tx, ty_min, ty_max, n_ty = angle_params
    # det_angles are tan_x,tan_y, the direction is (t_x,t_y,1)/norm
    det_angles = list(itertools.product(np.linspace(tx_min, tx_max, n_tx), np.linspace(ty_min, ty_max, n_ty)))
    det_angles = np.hstack((det_angles, np.ones((len(det_angles), 1))))
    det_angles = det_angles / np.linalg.norm(det_angles, axis=1, keepdims=True)
    if type(location_params) == tuple and len(location_params) == 7:
        x_min, x_max, n_x, y_min, y_max, n_y, z = location_params
        coords_list = list(
            map(np.array, itertools.product(np.linspace(x_min, x_max, n_x), np.linspace(y_min, y_max, n_y), [z])))
        if detector_shift is not None: coords_list = [c + detector_shift for c in coords_list]
        # coords_list = [c + np.random.rand(3) * 1.5 - 0.75 for c in coords_list]
    else:
        coords_list = list(map(np.array,location_params))
        if detector_shift is not None: coords_list = [c + detector_shift for c in coords_list]

    list_vox_coord = list_pred[:, :mass_loc]
    voxel_crosses = [[] for i in range(list_pred.shape[0])]
    detectors = []
    for i_d, coord in enumerate(coords_list):
        # vox_det = [voxels_per_direction(coord, ang, list_pred, size=size) for ang in det_angles]
        vox_det = []
        for i_a, ang in enumerate(det_angles):
            assert ang.size == 3
            vox_det.append(line_x_all_cubes(list_vox_coord, coord, ang, size / 2))
            if not vox_det[-1].size: continue
            for i_v in vox_det[-1]:
                voxel_crosses[i_v].append((i_d, i_a))
        # print(np.argwhere(vox_det))
        # vox_det = [[] for ang in det_angles]
        det_dict = {
            'coord': coord,
            'angles': det_angles,
            'dir_mass_true': np.array([passed_muons(dets, list_true, list_earth) for dets in vox_det]),
            'dir_mass_pred': np.array([passed_muons(dets, list_pred, list_earth) for dets in vox_det]),
            'vox_per_det_list': vox_det
        }
        detectors.append(det_dict)
    return detectors, voxel_crosses


def grad_step(n_step:int, lr:float, detector_list:list, vox_list:np.ndarray, vox_crosses:list, earth_list:np.ndarray = None, multi_det=0, eps=0.1,
              loss_function='l2', n_decay=5, verbose=True):
    """
    Performing a step for each voxel by adding a partial gradient of the loss function with possible constraints (number of detectors contributing to step in a voxel).

    :param n_step: id of the current step in the descent
    :param lr: learning rate (speed of the descent)
    :param detector_list: list of dictionaries with detectors ('coord','angles','dir_mass_true','dir_mass_pred')
    :param vox_list: (Nx*Ny*Nz,4) ndarray of voxel coordinates and masses (x,y,z,m)
    :param vox_crosses: list of lists, for each voxel is a list of (detector_id, direction_id) for all the rays that cross corresponding voxel
    :param multi_det: number of different detectors contributing to the anomaly in the voxel to include it in the step
    :param eps: minimal (abs) value of the anomaly to be added to the step
    :param loss_function: gradient of this function is calculated as a step. 'l1' - sum(abs(y-y')), 'l2' - sum((y-y')**2)
    :param n_decay: each n steps lr is divided by 2
    :param verbose: whether to print all unique voxel values every n_decay steps
    :return: updated detectors (pred part) and updated voxel list
    """
    fixed_cavities = True if vox_list.shape[1] > 4 else False
    mass_loc = -2 if fixed_cavities else -1
    assert (loss_function == 'l2' or loss_function == 'l1')
    # learning rate decay
    lr *= 0.5 ** np.floor((1 + n_step) / n_decay)
    for i_v in range(len(vox_list)):
        if not vox_crosses[i_v]: continue
        cross_v = np.array(vox_crosses[i_v])

        step_vox = 0  # gradient step
        det_v = set()  # detectors contributing to anomaly
        for i_d, i_l in cross_v:
            det = detector_list[i_d]
            mini_step = (det['dir_mass_true'][i_l] - det['dir_mass_pred'][i_l])
            if np.abs(mini_step) > eps:
                det_v |= {i_d}
                if loss_function == 'l2':
                    step_vox += 2 * mini_step
                elif loss_function == 'l1':
                    step_vox += mini_step / np.abs(mini_step)
        if multi_det and len(det_v) < multi_det: continue
        if fixed_cavities and not vox_list[i_v][-1]: continue
        vox_list[i_v][mass_loc] += lr * step_vox
    vox_list[:, mass_loc] = np.around(np.clip(vox_list[:, mass_loc], 0, 1), decimals=3)
    for det in detector_list:
        for i_l in range(len(det['angles'])):
            det['dir_mass_pred'][i_l] = passed_muons(det['vox_per_det_list'][i_l], vox_list, earth_list)
    if not (n_step % n_decay) and verbose: print(n_step, np.unique(np.around(vox_list[:, mass_loc], decimals=1)))
    return detector_list, vox_list


def visualise_detector(detector: dict, angle_params: tuple, i_det=0, iterated=True, inverse=False, origin='lower',
                       extent=(-1, 1, -1, 1)):
    """
    Plotting 2D angular histograms for a specific detector, true (experimental), simulated (predicted) and their bin-wise difference.

    :param detector: dictionary with all information about corresponding detector
    :param angle_params: min, max and n_steps for tan_x and tan_y angular binning
    :param i_det: detector id (for labeling the plot only)
    :param iterated: whether the simulation was fitted to represent experimental data
    :param origin: plt.imshow parameter, (0,0) location lower left/upper left
    :param extent: plt.imshow parameter, axes range
    :return: set of 2D histograms for the detector
    """
    sim = 'After processing: ' if iterated else 'Before  processing: '
    tx_min, tx_max, n_tx, ty_min, ty_max, n_ty = angle_params
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('Detector {} located at ({:.2f},{:.2f},{:.0f})'.format(i_det, *detector['coord']))
    # xy_tangents = (np.copy(detector['angles'])/(detector['angles'][:,-1])[:,np.newaxis])[:,:-1]
    # delta_tx, delta_ty = (tx_max-tx_min)/(n_tx-1), (ty_max-ty_min)/(n_ty-1)
    # xy_tangents[:, 0] /= delta_tx
    # xy_tangents[:, 1] /= delta_ty
    # xy_tangents -= xy_tangents.min(axis=0)
    # xy_tangents = np.array(xy_tangents, dtype=int)
    # hist_true, hist_pred = np.zeros((n_tx,n_ty)), np.zeros((n_tx,n_ty))
    # for i_ang, (i_x, i_y) in enumerate(xy_tangents):
    #     hist_true[i_x, i_y] = detector['dir_mass_true'][i_ang]
    #     hist_pred[i_x, i_y] = detector['dir_mass_pred'][i_ang]
    hist_true = np.reshape(detector['dir_mass_true'], (n_tx, n_ty)).copy()
    hist_pred = np.reshape(detector['dir_mass_pred'], (n_tx, n_ty)).copy()
    if inverse: hist_true *= -1; hist_pred *= -1
    mappa = cm.ScalarMappable(
        norm=colors.Normalize(vmin=min(hist_true.min(), hist_pred.min()), vmax=max(hist_true.max(), hist_pred.max())))
    ax1.imshow(hist_true, origin=origin, extent=extent)
    ax1.set_title('True')
    ax1.set_xlabel('tan_x')
    ax1.set_ylabel('tan_y')
    ax2.imshow(hist_pred, origin=origin, extent=extent)
    ax2.set_title(sim + 'Simulated')
    ax2.set_xlabel('tan_x')
    # ax2.set_ylabel('tan_y')
    fig.colorbar(mappa, ax=[ax1, ax2])
    mappa = cm.ScalarMappable(
        norm=colors.Normalize(vmin=(hist_true - hist_pred).min(), vmax=(hist_true - hist_pred).max()))
    ax3.imshow(hist_true - hist_pred, origin=origin, extent=extent)
    ax3.set_title('Difference')
    ax3.set_xlabel('tan_x')
    ax3.set_ylabel('tan_y')
    fig.colorbar(mappa, ax=ax3)
    plt.show()




list_init = voxel_array_to_list(volo_init, size=size)
list_true = voxel_array_to_list(volo_true, size=size)
list_earth = voxel_array_to_list(volo_earth, size=size)
detectors, voxel_crosses_list = create_detectors(angle_params, location_params, list_true, list_init, list_earth, size=size, detector_shift=shift)
print('initialization time:', datetime.now()-start)

# for i_d, det in enumerate(detectors):
i_d = 4; det = detectors[i_d]
visualise_detector(det, angle_params, i_det=i_d, iterated=False, inverse=True)
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
    detectors, list_pred = grad_step(s, lr, detectors, list_pred, voxel_crosses_list, earth_list=list_earth, multi_det=multi_detect, loss_function='l2', n_decay=10, verbose=False)

print('iteration time:', datetime.now()-start)
visualise_detector(detectors[i_d], angle_params, i_det=i_d, iterated=True, inverse=True)
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
# print('Error size (in wrong voxels) for {}x{} detectors: {:.2f}'.format(det_nx, det_nx, np.abs(volo_true - vox_pred_thresh)[...,0].sum()))

# Saving the results
# if os.path.exists('data/3d_output'): shutil.rmtree('data/3d_output')
# os.mkdir('data/3d_output')
# save_voxel_model(1-voxel_list_to_array(list_pred, size=size)[...,0].clip(0,1), format='.dat', file_name='data/3d_output/Model_simulated')
# for i_d, det in enumerate(detectors):
#     save_detector('data/detectors/det_final.txt', det, angle_params, i_d, sim_types=['pred'])

# for i_d in range(len(detectors)):
#     visualise_detector(detectors[i_d], angle_params, i_det=i_d, iterated=True)


# def rve_score(vox_true:np.ndarray, vox_init:np.ndarray, vox_pred:np.ndarray):
#     """
#     Remaining voxel error score: sum of voxel errors (number of wrong voxels) divided by number of initially wrong voxels.
#
#     :param vox_true: the ground-truth 3d model
#     :param vox_init: the starting 3d model
#     :param vox_pred: the 3d model after optimisation
#     :return: float, score
#     """
#     if len(vox_pred.shape)>3:
#         return np.sum(np.abs(vox_pred - vox_true)[...,0]) / np.sum(np.abs(vox_init - vox_true)[...,0])
#     return np.sum(np.abs(vox_pred-vox_true))/np.sum(np.abs(vox_init-vox_true))


print('Remaining voxel error: {:.2f}'.format(rve_score(volo_true, volo_init, vox_pred_thresh)))
