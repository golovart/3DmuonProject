import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools, gc, os
from datetime import datetime

# 3D functions


def line_x_cube(c, R=0.5, p1=None, d=None):
    """
    c - (x,y,z) of cube center
    R - half-size of the cube side
    p1,p2 - points defining the line
    """
    dim = len(c); eps  = 1e-6
    t = [sorted([(c[i]-R-p1[i])/(d[i]+eps), (c[i]+R-p1[i])/(d[i]+eps)]) for i in range(dim)]
    intersect = True
    for i,j in itertools.combinations(range(dim),2):
        intersect *= not (t[i][1]<t[j][0] or t[i][0]>t[j][1])
    return intersect


def passed_muons(det_coord, direction, voxel_list, size):
    """
    det_coord - (x,y,z) coordinates of detector
    direction - (tang_x, tang_y) defining from where the muons are coming
    voxel_list - voxels with masses (x,y,z,m), (0,0,0) is in the left down corner
    size - voxel (cube) size
    """
    mass_passed = 0.0
    #x0,y0,z0 = detector
    direction = (direction[0],direction[1],1)
    for *vox, mass in voxel_list:
        #vox -= det_coord
        if line_x_cube(vox, size/2, det_coord, direction): mass_passed += mass
    return mass_passed

def voxel_array_to_list(volume,size=1):
    vox_list = []
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            for k in range(volume.shape[2]):
                vox_list.append([(i+1/2)*size, (j+1/2)*size, (k+1/2)*size, volume[i,j,k]])
    return vox_list

# 2D functions

def passed_muons(det_coord, direction, voxel_list, size=1):
    """
    det_coord - (x,y) coordinates of detector
    direction - (tang_x) defining from where the muons are coming
    voxel_list - voxels with masses (x,y,m), (0,0) is in the left down corner
    size - voxel (cube) size
    """
    mass_passed = 0.0
    #x0,y0,z0 = detector
    if np.array(direction).size<2: direction = (direction,1)
    for *vox, mass in voxel_list:
        #vox -= det_coord
        if line_x_cube(vox, size/2, det_coord, direction):
            #print(vox)
            mass_passed += mass
    return mass_passed

def voxel_array_to_list(volume,size=1):
    vox_list = []
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
                vox_list.append([(i+1/2)*size, (j+1/2)*size, volume[i,j]])
    return vox_list

def crossed_voxels(detectors, voxel_list, size=1):
    #passed_muons(det_coord, direction, voxel_list, size=1):
    """
    detectors - list of dictionaries with detectors ('coord','angles','dir_mass_true','dir_mass_pred')
    voxel_list - voxels with masses (x,y,m), (0,0) is in the left down corner
    size - voxel (cube) size
    """
    voxel_crosses = [[] for i in range(len(voxel_list))]
#     mass_passed = 0.0
#     #x0,y0,z0 = detector
#     if np.array(direction).size<2: direction = (direction,1)
    for i_v, (*vox, mass) in enumerate(voxel_list):
        for i_d, det in enumerate(detectors):
            for i_l, direction in enumerate(det['angles']):
                if line_x_cube(vox, size/2, det['coord'], direction): voxel_crosses[i_v].append((i_d,i_l))
    return voxel_crosses

def grad_step(lr, detectors, vox_list, vox_crosses, size=1):
    """
    lr - learning rate
    detectors - list of dictionaries with detectors ('coord','angles','dir_mass_true','dir_mass_pred')
    voxel_list - voxels with masses (x,y,m), (0,0) is in the left down corner
    vox_crosses - list of (detector_id, direction_id) for all the lines that cross corresponding voxel
    size - voxel (cube) size
    """
    for i_v in range(len(vox_list)):
        # if voxel is crossed only by one detector - skip
        #if vox_crosses[i_v]:
        #    if len(np.unique(vox_crosses[i_v][:][0]))<2: continue
        for i_d, i_l in vox_crosses[i_v]:
            det = detectors[i_d]
            vox_list[i_v][-1] += 2*lr*(det['dir_mass_true'][i_l]-det['dir_mass_pred'][i_l])
    return vox_list

def voxel_list_to_array(vox_list, size=1):
    vox_list = np.array(vox_list)
    m, n = int(vox_list[:,0].max()/size+1/2), int(vox_list[:,1].max()/size+1/2)
    vox_arr = np.zeros((m,n))
    for v_i, v_j, m in vox_list:
        i, j = int(v_i/size-1/2), int(v_j/size-1/2)
        vox_arr[i,j] = m
    return vox_arr

def rot_angles(angles,th):
    matr = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    return np.matmul(matr, angles.T).T

volo_cube = np.zeros((100,100))
for i in range(volo_cube.shape[0]):
    for j in range(volo_cube.shape[1]):
        if (i>4 and i<95) and (j>4 and j<95): volo_cube[i,j] += 1
plt.imshow(volo_cube, alpha=0.6)

volo_hole = np.copy(volo_cube)
for i in range(volo_cube.shape[0]):
    for j in range(volo_cube.shape[1]):
        if ((i-49.5)**2+(j-49.5)**2<400): volo_hole[i,j] -= 1
plt.imshow(volo_hole, alpha=0.6)

volo_smallhole = np.copy(volo_hole)
for i in range(volo_cube.shape[0]):
    for j in range(volo_cube.shape[1]):
        if ((i-20.5)**2+(j-80.5)**2<50): volo_smallhole[i,j] -= 1
plt.imshow(volo_smallhole, alpha=0.6)
plt.show()

list_cube = voxel_array_to_list(volo_hole, size=1)
list_hole = voxel_array_to_list(volo_smallhole, size=1)

det_angles = np.array([(np.cos(a),np.sin(a)) for a in np.linspace(-np.pi/3,np.pi/3, 120)])

detectors = []
for i, coord in enumerate(itertools.product([-1],np.linspace(5,95,5))):
    if coord==(10,10): continue
    #angle = 225-45*np.abs(i-4)
    angle = 90
    if angle>90: angle -= 180
    det_dict = {
        'coord': coord[::-1],
        'angles': rot_angles(det_angles, angle*np.pi/180),
        'dir_mass_true': np.zeros(det_angles.shape[0]),
        'dir_mass_pred': np.zeros(det_angles.shape[0])
    }
    print(coord, angle)
    detectors.append(det_dict)

for det in detectors:
    #print(det)
    for i, l in enumerate(det['angles']):
        det['dir_mass_true'][i] = passed_muons(det['coord'], l, list_hole, size=1)
        det['dir_mass_pred'][i] = passed_muons(det['coord'], l, list_cube, size=1)

for det in detectors:
    c = det['coord']
    #if c!=(5,11): continue
    plt.plot(c[1]-0.5,c[0]-0.5,'o')
    for x,y in det['angles']:
        plt.plot(y*np.linspace(-100,300,300)+c[1]-0.5, x*np.linspace(-100,300,300)+c[0]-0.5, '--', alpha=0.6)
    #break
plt.imshow(volo_smallhole, alpha=0.5)
plt.imshow(volo_hole, alpha=0.5)
plt.show()


def grad_step(lr, detectors, vox_list, vox_crosses, size=1, multi_det=0, eps=0.1, deb=False):
    """
    lr - learning rate
    detectors - list of dictionaries with detectors ('coord','angles','dir_mass_true','dir_mass_pred')
    voxel_list - voxels with masses (x,y,m), (0,0) is in the left down corner
    vox_crosses - list of (detector_id, direction_id) for all the lines that cross corresponding voxel
    size - voxel (cube) size
    multi_det - include only anomalies with non-zero contributions from (multiple) detectors
    """
    n_det = len(detectors)
    if deb:
        i_v = 4055
        if not vox_crosses[i_v]: return vox_list
        cross_v = np.array(vox_crosses[i_v])

        step_v = 0  # gradient step
        det_v = set()  # detectors contributing to anomaly
        for i_d, i_l in cross_v:
            print(i_v, '\t', i_d, i_l)
            det = detectors[i_d]
            if np.abs(det['dir_mass_true'][i_l] - det['dir_mass_pred'][i_l]) > eps:
                det_v |= {i_d}
                step_v += (det['dir_mass_true'][i_l] - det['dir_mass_pred'][i_l])
            print(step_v)
        if multi_det and len(det_v) < multi_det: return vox_list
        vox_list[i_v][-1] += 2 * lr * step_v
        return vox_list

    for i_v in range(len(vox_list)):
        # if voxel is crossed by less than 2 detectors - skip
        if not vox_crosses[i_v]: continue
        cross_v = np.array(vox_crosses[i_v])

        step_v = 0  # gradient step
        det_v = set()  # detectors contributing to anomaly
        for i_d, i_l in cross_v:
            if deb: print(i_v, '\t', i_d, i_l)
            det = detectors[i_d]
            if np.abs(det['dir_mass_true'][i_l] - det['dir_mass_pred'][i_l]) > eps:
                det_v |= {i_d}
                step_v += (det['dir_mass_true'][i_l] - det['dir_mass_pred'][i_l])
            if deb: print(step_v)
        if multi_det and len(det_v) < multi_det: continue
        vox_list[i_v][-1] += 2 * lr * step_v  # (det['dir_mass_true'][i_l]-det['dir_mass_pred'][i_l])
    return vox_list


start = datetime.now()
num_steps = 15
lr = 0.001
# list_cube = voxel_array_to_list(volo_hole, size=1)
# for det in detectors:
#     for i, l in enumerate(det['angles']):
#         det['dir_mass_pred'][i] = passed_muons(det['coord'], l, list_cube, size=1)
crosses = crossed_voxels(detectors, list_cube, size=1)
for s in range(num_steps):
    list_cube = grad_step(lr, detectors, list_cube[:], crosses, size=1, multi_det=3, deb=False)
    for det in detectors:
        for i, l in enumerate(det['angles']):
            det['dir_mass_pred'][i] = passed_muons(det['coord'], l, list_cube, size=1)

plt.figure(figsize=(10,10))
plt.imshow(voxel_list_to_array(list_cube, size=1), alpha=0.8)
plt.colorbar();
plt.imshow(volo_smallhole, alpha=0.3)
plt.show()

print('iteration time:', datetime.now()-start)
