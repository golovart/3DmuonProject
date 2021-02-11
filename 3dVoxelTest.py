import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# # prepare some coordinates
# x, y, z = np.indices((8, 8, 8))
#
# # draw cuboids in the top left and bottom right corners, and a link between them
# cube1 = (x < 3) & (y < 3) & (z < 3)
# cube2 = (x >= 5) & (y >= 5) & (z >= 5)
# link = abs(x - y) + abs(y - z) + abs(z - x) <= 2
#
# # combine the objects into a single boolean array
# voxels = cube1 | cube2 | link
#
# volo_cube = np.zeros((20,20,20))
# for i in range(volo_cube.shape[0]):
#     for j in range(volo_cube.shape[1]):
#         for k in range(volo_cube.shape[2]):
#             if (i and i<19) and (j and j<19) and (k and k<19): volo_cube[i,j,k] = 1
# for i in range(volo_cube.shape[0]):
#     for j in range(volo_cube.shape[1]):
#         for k in range(volo_cube.shape[2]):
#             if ((i-9.5)**2+(j-9.5)**2+(k-9.5)**2<20): volo_cube[i,j,k] = 0
# for i in range(volo_cube.shape[0]):
#     for j in range(volo_cube.shape[1]):
#         for k in range(volo_cube.shape[2]):
#             if ((i-4.5)**2+(j-4.5)**2+(k-15.5)**2<5): volo_cube[i,j,k] = 0.5
#
#
# # set the colors of each object
# colors = np.zeros(volo_cube.shape+(4,))
# colors[...,0] = 1-np.abs(volo_cube-0.5)*2
# # colors[cube2] = 'green'
# # colors[...,2] = volo_cube
# colors[...,-1] = 0.5
#
#
# # and plot everything
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.voxels(volo_cube, edgecolor='k', facecolors=colors)#, alpha=0.5
# ax.plot(range(20), range(20), range(20), 'r')
#
# plt.show()


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def shrink(data):
    x, y, z = np.indices(np.array(data.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.02
    y[:, 0::2, :] += 0.02
    z[:, :, 0::2] += 0.02
    x[1::2, :, :] += 0.98
    y[:, 1::2, :] += 0.98
    z[:, :, 1::2] += 0.98
    return x, y, z


# build up the numpy logo
# n_voxels = np.zeros((4, 3, 4), dtype=bool)
# n_voxels[0, 0, :] = True
# n_voxels[-1, 0, :] = True
# n_voxels[1, 0, 2] = True
# n_voxels[2, 0, 1] = True
# facecolors = np.where(n_voxels, '#FFD65DC0', '#7A88CCC0')
# edgecolors = np.where(n_voxels, '#BFAB6E', '#7D84A6')
# filled = np.ones(n_voxels.shape)


volo_cube = np.ones((20, 20, 20))-0.05#, dtype=bool)
for i in range(volo_cube.shape[0]):
    for j in range(volo_cube.shape[1]):
        for k in range(volo_cube.shape[2]):
            if (i - 9.5) ** 2 + (j - 9.5) ** 2 + (k - 9.5) ** 2 < 20: volo_cube[i, j, k] = 0.1
            if (i - 4.5) ** 2 + (j - 4.5) ** 2 + (k - 15.5) ** 2 < 5: volo_cube[i, j, k] = 0.1
facecolors = cm.inferno(1-volo_cube)
facecolors[...,-1] = 1-volo_cube
# facecolors = np.where(volo_cube[...,np.newaxis], (221,19,19,0.9), (255,214,93,0.2))
#edgecolors = np.where(volo_cube, '#BFAB6E', '#7D84A6')
filled = np.ones(volo_cube.shape)

# upscale the above voxel image, leaving gaps
filled_2 = explode(filled)
fcolors_2 = explode(facecolors)
#ecolors_2 = explode(edgecolors)

# Shrink the gaps
# x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
# x[0::2, :, :] += 0.01
# y[:, 0::2, :] += 0.01
# z[:, :, 0::2] += 0.01
# x[1::2, :, :] += 0.99
# y[:, 1::2, :] += 0.99
# z[:, :, 1::2] += 0.99

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(*shrink(filled_2), filled_2, facecolors=fcolors_2)#, edgecolors=ecolors_2, alpha=0.5)

plt.show()
