import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import itertools, gc, os, h5py, shutil, sys


def process_arguments(argument_list):
    """
    Process command line arguments of the script, according to notations in 'help' (-h).
    If no arguments passed, default parameter values are returned.

    :param argument_list: sys.argv instance
    :return:
    Simulation parameters:
    input_dir:dict - dict('path':str, 'read':bool, 'save':bool) for processing the input
    output_dir:dict - dict('path':str, 'read':bool, 'save':bool) for processing the output
    det_nx:int, det_ny:int - size of detector grid N1xN2
    scene_shape:tuple - N1xN2xN3 shape of the voxel structure
    num_steps:int - number of gradient steps
    lr:float - learning rate
    thresh:float - anomaly threshold applied after all iterations
    reiterate:bool - whether to run the iterations (grad steps), or only load/save/visualise the data
    det_id:int - id of the detector to visualise in 2D angular histograms
    visualise:dict - dict('det':bool, 'vox':bool) for visualising detectors and 3d models
    """
    input_dir = {'path':None, 'read':False, 'save':False}
    output_dir = {'path':None, 'read':False, 'save':False}
    det_nx, det_ny = 3, 3
    scene_shape = (32,32,32)
    num_steps = 50
    lr = 0.06 / det_nx / scene_shape[2]
    thresh = 0.4
    det_id = 1
    reiterate = True
    visualise = {'det': False, 'vox': False}
    if not len(argument_list)>1:
        return input_dir, output_dir, det_nx, det_ny, scene_shape, num_steps, lr, thresh, reiterate, det_id, visualise
    else: argument_list = argument_list[1:]

    if not argument_list[0].startswith('-'): raise KeyError('Undefined argument: '+argument_list[0])

    known_keys = ['h','i','o','s','d','n','l','t','ni','v']
    help_msg = """usage: python 3dSimulation.py [opt1] [arg1] [opt2] [arg2] ...
    -h  \t: print help message
    -i dir\t: input directory to load input (optional). If not provided, default scene is created and not saved in files. 
    \t\t  If provided, but empty, default scene is created and saved in the directory. 
    -o dir\t: output directory (optional). If not provided, output files are not saved. If not empty, recreated as empty.
    -d N\t: size of the detector grid. If 1 number is passed, grid is NxN. If 2 numbers are passed, grid is N1xN2.
    -s S\t: voxel size of the scene. If 1 number is passed, 3d array's shape is SxSxS. If 3 numbers are passed, shape is S1xS2xS3
    -n num\t: number of gradient steps. If not provided, default is 50
    -l lr\t: learning rate. If not provided, default is calculated from number of detectors and voxel shape, close to 0.001
    -t thr\t: threshold on the voxel value for the final 3d model. If not provided, default is 0.4
    -ni \t: not-iterate. If provided, the iteration is not performed, input and/or output are loaded from directories.
    -v [det [id]] [vox] : whether to visualise the output. If 'det' key provided, the id detector is visualised 
    \t\t\t   (if id not provided, id=1). If 'vox' key provided, 3d voxel models are visualised using matplotlib.
    """
    arguments = {}
    key = 'u'
    for arg in argument_list:
        if arg.startswith('-'):
            key = arg[1:]
            if key not in known_keys: raise KeyError('Undefined argument: -' + key)
            arguments[key] = []
        else:
            arguments[key].append(arg)
    if 'u' in arguments.keys(): raise KeyError('Undefined argument: '+argument_list[0])

    if 'h' in arguments.keys():
        print(help_msg)
        exit()
    if 'i' in arguments.keys():
        if len(arguments['i']) < 1: raise ValueError('Missing argument for input dir')
        input_dir['path'] = arguments['i'][0]
        if len(arguments['i']) > 1: raise ValueError('Too many arguments for input dir')
        if os.path.exists(input_dir['path']):
            if len(os.listdir(input_dir['path'])):
                input_dir['read'] = True
            else:
                input_dir['save'] = True
        else:
            os.mkdir(input_dir['path'])
            input_dir['save'] = True
    if 'o' in arguments.keys():
        if len(arguments['o']) < 1: raise ValueError('Missing argument for output dir')
        output_dir['path'] = arguments['o'][0]
        if len(arguments['o']) > 1: raise ValueError('Too many arguments for output dir')
        if os.path.exists(output_dir['path']):
            if len(os.listdir(output_dir['path'])):
                output_dir['read'] = True
            else:
                output_dir['save'] = True
        else:
            os.mkdir(output_dir['path'])
            output_dir['save'] = True
    if 'd' in arguments.keys():
        if len(arguments['d']) == 1: det_nx, det_ny = int(arguments['d'][0]), int(arguments['d'][0])
        elif len(arguments['d']) == 2: det_nx, det_ny = int(arguments['d'][0]), int(arguments['d'][1])
        else: raise ValueError('Wrong arguments for the detector grid size')
    if 's' in arguments.keys():
        if len(arguments['s']) == 1: scene_shape = (int(arguments['s'][0]),int(arguments['s'][0]),int(arguments['s'][0]))
        elif len(arguments['s']) == 3: scene_shape = (int(arguments['s'][0]),int(arguments['s'][1]),int(arguments['s'][2]))
        else: raise ValueError('Wrong arguments for the voxel model size')
        lr = 0.06 / det_nx / scene_shape[2]
    if 'n' in arguments.keys():
        if len(arguments['n']) == 1: num_steps = int(arguments['n'][0])
        else: raise ValueError('Wrong arguments for the number of gradient steps')
    if 'l' in arguments.keys():
        if len(arguments['l']) == 1: lr = float(arguments['l'][0])
        else: raise ValueError('Wrong arguments for the learning rate')
    if 't' in arguments.keys():
        if len(arguments['t']) == 1: thresh = float(arguments['t'][0])
        else: raise ValueError('Wrong arguments for the threshold')
    if 'ni' in arguments.keys():
        reiterate = False
    if 'v' in arguments.keys():
        if not len(arguments['v']): raise ValueError('Missing argument for visualisation')
        if 'det' in arguments['v']: visualise['det'] = True
        if 'vox' in arguments['v']: visualise['vox'] = True
        if 'det' in arguments['v'] and arguments['v'][-1]!='det':
            det_id = ([int(arguments['v'][i+1]) for i,arg in enumerate(arguments['v']) if arg=='det'])[0]
    return input_dir, output_dir, det_nx, det_ny, scene_shape, num_steps, lr, thresh, reiterate, det_id, visualise


def add_empty_sphere(vox_array: np.ndarray, c: np.ndarray, r: float, delta=0.01):
    """
    Adds a sphere filled with passed value

    :param vox_array: initial voxel volume
    :param c: (x,y,z) sphere's center coordinates
    :param r: sphere's radius
    :param delta: a value to be assigned to the voxels inside the sphere
    :return: updated vox_array
    """
    for i in range(vox_array.shape[0]):
        for j in range(vox_array.shape[1]):
            for k in range(vox_array.shape[2]):
                if (i - c[0]) ** 2 + (j - c[1]) ** 2 + (k - c[2]) ** 2 < r ** 2: vox_array[i, j, k] = delta
    return vox_array


def add_empty_tunnel(vox_array: np.ndarray, start: np.ndarray, end: np.ndarray, r: float, delta=0.01):
    """
    Adds a tunnel (cylinder with half spheres on ends) filled with passed value

    :param vox_array: initial voxel volume
    :param start: (x,y,z) coordinates of the 'bottom' center of the cylinder
    :param end: (x,y,z) coordinates of the 'top' center of the cylinder
    :param r: radius of the cylinder and half-spheres
    :param delta: value to be filled in the corresponding voxels
    :return: updated vox_array
    """
    l12 = end - start;
    d1 = -np.dot(l12, start);
    d2 = -np.dot(l12, end)
    for i in range(vox_array.shape[0]):
        for j in range(vox_array.shape[1]):
            for k in range(vox_array.shape[2]):
                p = np.array([i, j, k]);
                l = p - start
                if (np.linalg.norm(np.cross(l, l12)) / np.linalg.norm(l12) < r and np.dot(l12, p) + d2 < 0 and np.dot(
                        l12, p) + d1 > 0) or np.linalg.norm(l) < r or np.linalg.norm(p - end) < r:
                    vox_array[i, j, k] = delta
    return vox_array


def create_default_voxel_setting(scene_shape=(20,20,20), delta_alpha=1.0, shift: np.ndarray = 0, size=1.0):
    """
    Creating 'true' and 'initial' voxel arrays of complex predefined configuration scaled to required shape.
    If a 4-dimensional shape is given (then last dimension has to be 2), initial cavities are fixed, every voxel
    contains value and 0/1 defining whether it's fixed.

    :param scene_shape: tuple of shape (preferably square), either (n,n,n) or (n,n,n,2)
    :param delta_alpha: value for cavities to be filled with (rest is 1-detla_aplha)
    :param shift: a 3d-vector to shift whole structure, default=0
    :param size: voxel size (>1 can be used when merging voxels into bigger voxels) [currently not implemented]
    :return: voxel_init and voxel_true - corresponding np.ndarrays
    """
    scale = scene_shape[2] / 20

    s1, s2, s3, s4, s5 = np.array((3.5, 3.5, 5.5)), np.array((14.5, 9.5, 15.5)), np.array((12.5, 2.5, 2.5)), np.array(
        (9.5, 15.5, 3.5)), np.array((9, 6.5, 10.5))
    if shift is not None:
        s1 += shift;        s2 += shift;        s3 += shift;        s4 += shift;        s5 += shift
    s1 *= scale;    s2 *= scale;    s3 *= scale;    s4 *= scale;    s5 *= scale
    r1, r2, r3, r4, rl = 1.5, 2.5, 2, 2, 1
    r1 *= scale;    r2 *= scale;    r3 *= scale;    r4 *= scale;    rl *= scale
    volo_init = np.ones(scene_shape)
    delta_init = delta_alpha if len(scene_shape)<4 else (delta_alpha,0)
    delta_true = delta_alpha if len(scene_shape)<4 else (delta_alpha,1)
    if len(scene_shape)<4:
        volo_init -= delta_alpha
    else:
        volo_init[...,0] -= delta_alpha
    volo_init = add_empty_sphere(volo_init, s1, r1, delta_init)
    volo_init = add_empty_tunnel(volo_init, s1, s2, rl, delta_init)
    volo_init = add_empty_sphere(volo_init, s2, r2, delta_init)
    volo_init = add_empty_tunnel(volo_init, s2, s3, rl, delta_init)
    volo_init = add_empty_sphere(volo_init, s3, r3, delta_init)
    volo_init = add_empty_tunnel(volo_init, s3, s5, rl, delta_init)
    volo_init = add_empty_tunnel(volo_init, s2, s4, rl, delta_init)
    volo_init = add_empty_sphere(volo_init, s4, r4, delta_init)

    volo_true = np.copy(volo_init)
    volo_true = add_empty_sphere(volo_true, (np.array((15.5, 13.5, 3.5)) + shift) * scale, 2.5 * scale,
                                 delta_true)
    volo_true = add_empty_sphere(volo_true, (np.array((12, 5, 15.5)) + shift) * scale, 1.8 * scale, delta_true)
    volo_true = add_empty_sphere(volo_true, (np.array((4, 8, 5.5)) + shift) * scale, 2 * scale, delta_true)
    return volo_init, volo_true


# 3D functions
def passed_muons(passed_list: list, voxel_list: np.ndarray):
    """
    Calculating amount of 'mass' passed by muons in specific direction (ray)

    :param passed_list: list of voxel ids crossed by this ray
    :param voxel_list: list of voxels (x,y,z,m)
    :return: float
    """
    mass_loc = -2 if voxel_list.shape[1] > 4 else -1
    # fixed_cavities = True if voxel_list.shape[1]>4 else False
    mass_passed = 0.0
    for i_v in passed_list:
        mass_passed += voxel_list[i_v][mass_loc]
    return mass_passed


def voxel_array_to_list(volume: np.ndarray, size=1):
    """
    Convert 3D voxel volume into a list of voxel coordinates and values (x,y,z,m)

    :param volume: 3D-array of voxel masses
    :param size: voxel size (>1 can be used when merging voxels into bigger voxels)
    :return: list of (x,y,z,m) converted to (N*N*N,4) ndarray
    """
    fixed_cavities = True if len(volume.shape) > 3 else False
    vox_list = []
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            for k in range(volume.shape[2]):
                if fixed_cavities:
                    vox_list.append([(i + 1 / 2) * size, (j + 1 / 2) * size, (k + 1 / 2) * size, *volume[i, j, k]])
                else:
                    vox_list.append([(i + 1 / 2) * size, (j + 1 / 2) * size, (k + 1 / 2) * size, volume[i, j, k]])
    return np.array(vox_list)


def voxel_list_to_array(vox_list: np.ndarray, size=1):
    """
    Convert a (Nx*Ny*Nz,3) list into (Nx,Ny,Nz) ndarray of masses

    :param vox_list: (Nx*Ny*Nz,4) ndarray of voxel coordinates and masses (x,y,z,m)
    :param size: voxel size (>1 can be used when merging voxels into bigger voxels)
    :return: (Nx,Ny,Nz) ndarray
    """
    fixed_cavities = True if vox_list.shape[1] > 4 else False
    m, n, l = int(vox_list[:, 0].max() / size + 1 / 2), int(vox_list[:, 1].max() / size + 1 / 2), int(
        vox_list[:, 2].max() / size + 1 / 2)
    vox_arr = np.zeros((m, n, l)) if not fixed_cavities else np.zeros((m, n, l, 2))
    for v_i, v_j, v_k, *mass in vox_list:
        i, j, k = int(v_i / size - 1 / 2), int(v_j / size - 1 / 2), int(v_k / size - 1 / 2)
        vox_arr[i, j, k] = mass if fixed_cavities else mass[0]
    return vox_arr


def voxels_vs_detectors(voxel_list: np.ndarray, detectors: dict, size=1):
    """
    Find correspondence between voxels and rays crossing them.

    :param voxel_list: (Nx*Ny*Nz,4) ndarray of voxel coordinates and masses (x,y,z,m)
    :param detectors: dict (with i_d as keys) of dictionaries with detectors ('coord','angles','dir_mass_true','dir_mass_pred')
    :param size: voxel size (>1 can be used when merging voxels into bigger voxels)
    :return:
    detectors: dict of dictionaries with ('coord','angles','dir_mass_true','dir_mass_pred')
    vox_crosses: list of lists, for each voxel is a list of (detector_id, direction_id) for all the rays that cross corresponding voxel
    """
    mass_loc = -2 if voxel_list.shape[1] > 4 else -1
    list_vox_coord = voxel_list[:, :mass_loc]
    voxel_crosses = [[] for i in range(voxel_list.shape[0])]
    for i_d in detectors.keys():
        vox_det = []
        for i_a, ang in enumerate(detectors[i_d]['angles']):
            assert ang.size == 3
            vox_det.append(line_x_all_cubes(list_vox_coord, detectors[i_d]['coord'], ang, size / 2))
            if not vox_det[-1].size: continue
            for i_v in vox_det[-1]:
                voxel_crosses[i_v].append((i_d, i_a))
        detectors[i_d]['vox_per_det_list'] = vox_det
    return detectors, voxel_crosses


def line_x_all_cubes(cube_list: np.ndarray, p1: np.ndarray, d: np.ndarray, r=0.5):
    """
    Find ids of voxels that intersect with a specific direction (ray).
    For optimisation, the crossing of the spheres with same radius, located in the voxel centers, is considered instead of cubes.

    :param cube_list: (Nx*Ny*Nz,3) ndarray of voxel coordinates (x,y,z)
    :param p1: detector coordinates
    :param d: normalised direction vector
    :param r: cube (sphere) radius
    :return: 1darray of indices
    """
    return np.argwhere(np.abs(np.linalg.norm(np.cross(cube_list - p1, d), axis=1)) < r).ravel()


def create_detectors(angle_params: tuple, location_params: tuple, list_true: np.ndarray, list_pred: np.ndarray, size=1.0, detector_shift=None):
    """
    Creates a set of detectors with defined parameters (angular binning and location coordinates) and finds correspondence between voxels and directions (rays) crossing them for every detector.
    Calculates masses along each direction for true and (initial) predicted 3d models.

    :param angle_params: min, max and n_steps for tan_x and tan_y angular binning
    :param location_params: (x_min, x_max, n_x, y_min, y_max, n_y, z) or tuple (len!=7) of (x,y,z) for each detector
    :param list_true: real 3d model, (Nx*Ny*Nz,4) ndarray of voxel coordinates and masses (x,y,z,m)
    :param list_pred: predicted 3d model, (Nx*Ny*Nz,4) ndarray of voxel coordinates and masses (x,y,z,m)
    :param size: voxel size, default is 1
    :param detector_shift: (x,y,z), if not None all detectors have to be shifted in fixed direction
    :return:
    detectors: dict of dictionaries with ('coord','angles','dir_mass_true','dir_mass_pred')
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
    detectors = {}
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
            'angle_params': angle_params,
            'dir_mass_true': np.array([passed_muons(dets, list_true) for dets in vox_det]),
            'dir_mass_pred': np.array([passed_muons(dets, list_pred) for dets in vox_det]),
            'vox_per_det_list': vox_det
        }
        detectors[i_d] = det_dict
    return detectors, voxel_crosses


def grad_step(n_step:int, lr:float, detector_list:dict, vox_list:np.ndarray, vox_crosses:list, multi_det=0, eps=0.1,
              loss_function='l2', n_decay=5, verbose=True):
    """
    Performing a step for each voxel by adding a partial gradient of the loss function with possible constraints (number of detectors contributing to step in a voxel).

    :param n_step: id of the current step in the descent
    :param lr: learning rate (speed of the descent)
    :param detector_list: dict (with i_d as keys) of dictionaries with detectors ('coord','angles','dir_mass_true','dir_mass_pred')
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
    for i_d in detector_list.keys():
        for i_l in range(len(detector_list[i_d]['angles'])):
            detector_list[i_d]['dir_mass_pred'][i_l] = passed_muons(detector_list[i_d]['vox_per_det_list'][i_l], vox_list)
    if not (n_step % n_decay) and verbose: print(n_step, np.unique(np.around(vox_list[:, mass_loc], decimals=1)))
    return detector_list, vox_list


def explode(data: np.ndarray):
    """
    Add empty psuedo-voxels between original for visualisation of all voxel sides.
    (Nx,Ny,Nz)->(2Nx,2Ny,2Nz)

    :param data: 3d array of numbers
    :return: enlarged 3d array
    """
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def shrink(data: np.ndarray):
    """
    Shrink the pseudo-voxel coordinates produced in "explode", so the sides of real voxels will be close to each other.

    :param data: 'exploded' 3d array
    :return: tuple of 3d arrays of coordinates for correct visualisation of the voxels (shrinking the pseudo-voxels)
    """
    x, y, z = np.indices(np.array(data.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.02
    y[:, 0::2, :] += 0.02
    z[:, :, 0::2] += 0.02
    x[1::2, :, :] += 0.98
    y[:, 1::2, :] += 0.98
    z[:, :, 1::2] += 0.98
    return x, y, z


def visualise_voxels(vox_cubes: np.ndarray, inv=True, figsize=(5, 5), det_coords=None):
    """
    Plot a 3D visualization of the voxels. Intensities are from Inferno colormap. Transparency is equal to value in the voxel.
    Inverted image is (1-original). If detector coordinates are provided, they are added as a scatter plot.

    :param vox_cubes: 3d array of voxels
    :param inv: bool whether to invert filled-empty
    :param figsize: matplotlib figure size
    :param det_coords: 2d array (n_det, 3) of detector coordinates
    :return: (interactive) 3d graph
    """
    facecolors = cm.inferno(inv + ((-1) ** inv) * vox_cubes)
    facecolors[..., -1] = inv + ((-1) ** inv) * vox_cubes
    filled = np.ones(vox_cubes.shape)
    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.voxels(*shrink(filled_2), filled_2, facecolors=fcolors_2)
    if det_coords is not None:
        ax.scatter(det_coords[:, 0], det_coords[:, 1], zs=np.min(det_coords[:, 2]), zdir='z', linewidth=3)
    plt.show(block=False)


def visualise_detector(detector: dict, i_det=0, iterated=True, origin='lower',
                       extent=(-1, 1, -1, 1)):
    """
    Plotting 2D angular histograms for a specific detector, true (experimental), simulated (predicted) and their bin-wise difference.

    :param detector: dictionary with all information about corresponding detector
    :param i_det: detector id (for labeling the plot only)
    :param iterated: whether the simulation was fitted to represent experimental data
    :param origin: plt.imshow parameter, (0,0) location lower left/upper left
    :param extent: plt.imshow parameter, axes range
    :return: set of 2D histograms for the detector
    """
    sim = 'After processing: ' if iterated else 'Before  processing: '
    tx_min, tx_max, n_tx, ty_min, ty_max, n_ty = detector['angle_params']
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
    hist_true = np.reshape(detector['dir_mass_true'], (n_tx, n_ty))
    hist_pred = np.reshape(detector['dir_mass_pred'], (n_tx, n_ty))
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
    plt.show(block=False)


def rve_score(vox_true:np.ndarray, vox_init:np.ndarray, vox_pred:np.ndarray):
    """
    Remaining voxel error score: sum of voxel errors (number of wrong voxels) divided by number of initially wrong voxels.

    :param vox_true: the ground-truth 3d model
    :param vox_init: the starting 3d model
    :param vox_pred: the 3d model after optimisation
    :return: float, score
    """
    if len(vox_pred.shape)>3:
        return np.sum(np.abs(vox_pred - vox_true)[...,0]) / np.sum(np.abs(vox_init - vox_true)[...,0])
    return np.sum(np.abs(vox_pred-vox_true))/np.sum(np.abs(vox_init-vox_true))


def save_voxel_model(voxels: np.ndarray, format='.txt', file_name='voxel_model', data_name='predicted_model', size=1):
    """
    Save 3d voxel model either as a ndarray (in hdf5), a list of (x,y,z,m) voxels (in txt) or a flattened array of bytes (binary format).
    In binary format first 24 bins are 3 integers - shape of the voxel array (Nx,Ny,Nz), rest is np.float32 in bytes.

    :param voxels: 3D array of masses
    :param format: data format '.h5'/'.txt'/'.dat'
    :param file_name: main part of the file name, format is appended in the end
    :param data_name: the dataset name, (only) if saving in hdf5
    :param size: voxel size, default is 1
    :return:
    """
    if format == '.h5':
        file_name += format
        with h5py.File(file_name, 'w') as outf:
            outf.create_dataset(data_name, data=voxels)
    elif format == '.txt':
        file_name += format
        with open(file_name, 'w') as outf:
            outf.write('{},{},{}\n'.format(size, size, size))
            nx, ny, nz = voxels.shape
            outf.write('{},{},{}\n'.format(nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        outf.write('{},{},{},{}\n'.format(i, j, k, voxels[i, j, k]))
            # for *vox, m in voxels:
            #     # if 1-m<delta*10: continue
            #     outf.write('{},{},{},{}\n'.format(*(np.array(vox)/size - 1/2).astype(int), m))
    else:
        file_name += '_' + 'x'.join(map(str, voxels.shape))
        file_name += format
        with open(file_name,'wb') as outf:
            outf.write(np.array(voxels.shape).tobytes())
            voxels.astype(dtype=np.float32).ravel().tofile(outf)  # +'.np')


def read_voxel_model(file_name: str, data_name='predicted_model'):
    """
    Read 3d array of masses from file.

    :param file_name: path to file with extension '.h5'/'.txt'/'.dat'
    :param data_name: the dataset name, (only) if loading from hdf5
    :return:  3D voxel array
    """
    if file_name.split('.')[-1] == 'h5':
        with h5py.File(file_name, 'r') as df:
            voxels = df[data_name][...]
        return voxels
    elif file_name.split('.')[-1] == 'txt':
        with open(file_name, 'r') as df:
            file_lines = df.readlines()
            size = eval(file_lines[0])
            if not all([size[i] == size[i + 1] for i in range(len(size) - 1)]):
                print('Implement non-cubic voxels')
                return False
            size = size[0]
            nx, ny, nz = eval(file_lines[1])
            voxels = np.zeros((nx, ny, nz))
            for vox_line in file_lines[2:]:
                i, j, k, m = eval(vox_line)
                voxels[i, j, k] = m
            return voxels
    else:
        with open(file_name, 'rb') as df:
            byte_data = df.read()
            shape =np.frombuffer(byte_data[:24], dtype=np.int)
            voxels = np.frombuffer(byte_data[24:], dtype=np.float32).reshape(shape).copy()
        return voxels


def save_detector(file_name: str, detector: dict, i_det=0, sim_types: list = None):
    """
    Save a 2d angular histogram of the detector.

    :param file_name: path to file
    :param detector: dictionary with all information about corresponding detector
    :param i_det: detector id (for labeling the plot only)
    :param sim_types: which data to save: 'true'(experimental), 'pred'(simulated)
    :return:
    """
    file_name = file_name.split('.')[0] + '_' + str(i_det)
    tx_min, tx_max, n_tx, ty_min, ty_max, n_ty = detector['angle_params']
    if sim_types is None: sim_types = ['true', 'pred']
    for sim_type in sim_types:
        with open(file_name + '_' + sim_type + '.txt', 'w') as outf:
            outf.write(str(i_det) + '\n')
            outf.write('{:.2f} {:.2f} {:.2f}\n'.format(*detector['coord']))
            outf.write('0 0 0\n')
            outf.write('{} {} {} {}\n'.format(tx_min, tx_max, ty_min, ty_max))
            outf.write('{} {}\n'.format(n_tx, n_ty))
            det_array = np.reshape(detector['dir_mass_' + sim_type], (n_tx, n_ty))
            np.savetxt(outf, det_array, fmt='%.2f', delimiter=' ')


def read_detectors(detector_dir: str, detector_state='init'):
    """
    Read detector parameters and 2d histograms from all files starting with 'det' in the directory.

    :param detector_dir: path to directory with detector files
    :param detector_state: detector state before or after simulation, can be 'init' or 'final'. "True" detector is always 'init'.
    :return:
    detectors: dict of dictionaries with ('coord','angles','dir_mass_true','dir_mass_pred')
    """
    det_list = [d for d in os.listdir(detector_dir) if d.startswith('det')]
    # det_types = np.unique([d.split('_')[1] for d in det_list])
    detectors = {}
    for det_name in det_list:
        if not detector_state == det_name.split('_')[1]: continue
        det_type = (det_name.split('_')[-1])[:-4]
        det_tmp = {}
        with open(detector_dir+'/'+det_name,'r') as df:
            det_tmp['id'] = eval(df.readline())
            det_tmp['coord'] = np.array(list(map(eval,df.readline().split(' '))))
            det_tmp['orient'] = np.array(list(map(eval,df.readline().split(' '))))
            tx_min, tx_max, ty_min, ty_max = list(map(eval,df.readline().split(' ')))
            n_tx, n_ty = list(map(eval,df.readline().split(' ')))
            # det_angles are tan_x,tan_y, the direction is (t_x,t_y,1)/norm
            det_angles = list(itertools.product(np.linspace(tx_min, tx_max, n_tx), np.linspace(ty_min, ty_max, n_ty)))
            det_angles = np.hstack((det_angles, np.ones((len(det_angles), 1))))
            det_tmp['angles'] = det_angles / np.linalg.norm(det_angles, axis=1, keepdims=True)
            det_tmp['angle_params'] = (tx_min, tx_max, n_tx, ty_min, ty_max, n_ty)
            det_tmp['dir_mass_'+det_type] = np.loadtxt(df).ravel()
        if det_tmp['id'] in detectors.keys():
            detectors[det_tmp['id']]['dir_mass_'+det_type] = det_tmp['dir_mass_'+det_type]
        else:
            detectors[det_tmp['id']] = det_tmp
    return detectors


def merge(vox_array, factor=2, delta=0.01):
    # Potential implementation of merging small voxels into larger ones for pre-study with identifying interesting areas
    big_shape = np.ceil(np.array(vox_array.shape)/factor).astype(int)
    vox_big = np.zeros(big_shape)
    for i in range(big_shape[0]):
        for j in range(big_shape[1]):
            for k in range(big_shape[2]):
                vox_big[i,j,k] = np.sum(vox_array[i*factor:(i+1)*factor, j*factor:(j+1)*factor, k*factor:(k+1)*factor])/factor**3
    # vox_big = np.where(vox_big>0.5, 1-delta, delta)
    # TODO: implement threshold before visualisation
    return vox_big
