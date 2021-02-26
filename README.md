# Gradient descent based 3D reconstruction from 2D angular projections for Muon radiography

In this project we want to reconstruct an approximate 3D voxel model of some object/structure,
for simplicity, consisting only of filled or empty voxels, or to improve the precision of 
the already existing 3D model. Reconstruction is performed with data from a set of detectors, 
represented by 2D angular projections - histograms, where each bin corresponds to filled (or empty) 
distance in that direction from detector to the Earth surface (boundary of the voxel model).

###### Input data
* Initial approximate 3D voxel model of the object (0 and 1 represent filled and empty voxels).
* (optional) True 3D voxel model (only for cross-checking the output)
* True (experimental) 2D angular histograms for all the detectors
* Initial approximate 2D angular histograms (optional, can be obtained from the approximate voxel model)
* simulation parameters: learning rate for gradient descent, scene shape (can be obtained from angular binning)

###### Output data
* optimized 2D angular histograms
* optimized 3D voxel model (voxel values in range [0,1.0], 
  but a threshold can be applied to obtain binary representation)
* Remaining voxel error score - sum of voxel errors (number of wrong voxels) divided by number of initially wrong voxels.

## Project structure

* _3dSimulation.py_ - main simulation script. Can take model and detector files 
  to perform a simulation or run with the default setting.
* _create_scene.py_ - script to create default scene and save into input files.
* _muon_utils.py_ - set of functions used in simulation: creating voxels, detectors, 
  performing gradient steps, visualising detectors and voxels, saving and reading detectors and models.
* _Sandbox.py_ - script for playing around with potential new features. 
* ___tests___ - folder with scripts for testing the performance of different configurations 
  of experiment and/or simulation
  * _detectors.py_ - comparing performance for different configurations and number of detectors.
  * _fixed_cavities.py_ - comparing results depending on whether initially known cavities are fixed during iterations (reduces amount of unknown parameters).
  * _multi_detectors.py_ - step is added to the voxel value only if there are contributions to the error from several detectors. This tests different minimum number of contributing detectors. 
  * _scale.py_ - comparing performance for the same spheres&tunnels setting, but for different scale: 16x16x16 - 64x64x64
  * *__results.txt_ - outputs of the corresponding tests. 
