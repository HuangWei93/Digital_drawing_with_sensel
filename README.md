# Digital drawing with sensel

## Structure
```
|-- aruco_markers
|-- collecting_textures
|   |-- animation_of_pixels
|   |-- camera_cal
|   |-- centroids
|   |-- curves
|   |-- images
|   |-- images_with_centroids
|   |-- images_with_resampled_patches
|   |-- pixels
|   |-- processed_images
|-- dataset
|   |-- animation_pixels.py
|   |-- classification
|   |   |-- merged_images
|   |   |-- paras
|   |   |-- textures
|   |-- image_with_patches
|-- generate_paras
|   |-- inputs
|   |-- modified_inputs
|-- reconstruct_stroke
|   |--reconstruct_stroke_with_eqdist_by_clustering_paras
|   |--reconstructed_strokes_with_eqdist_by_search
|-- Graphite-Pencil
```
## Hardware setup 
### Sensel Morph Tablet 
![](tablet.jpeg)
### Pencil with gyro and ESP board
![](pencil_with_gyro.jpeg)
## Collecting dataset
We are going to generate dataset:

<coordinate-x, coordinate-y, velocity-x, velocity-y, force, original_phi, theta, modified_phi>
### generate raw dataset
```
python generate_input image_id 
python frame.py
```
### process dataset
```
python modify_coords.py 
```
### crop images
```
python image_processing.py
```
### collecting textures
```
python construct_dataset.py size_of_windows overlap_rate
```
### classify textures by kmeans
```
python  kmeans_clustering_paras.py
python  kmeans_clustering_textures.py
```
###  reconstruct stroke
```
python reconstruct_stroke_with_eqdist_by_clustering_paras.py size_of_windows overlap_rate
python reconstruct_stroke_with_eqdist_by_search.py size_of_windows overlap_rate
```

## Graphite-Pencil Model
Compile main.cpp 
https://github.com/Aquietzero/Graphite-Pencil
