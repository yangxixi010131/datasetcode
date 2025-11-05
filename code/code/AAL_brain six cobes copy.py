import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, image

# Load functional connectivity matrices
correlation_matrices = np.load('all_correlation_matrices103.npy')

# Extract the first subject's functional connectivity matrix
first_subject_matrix = correlation_matrices[0]

# Load AAL atlas template
atlas = datasets.fetch_atlas_aal()
atlas_img = image.load_img(atlas.maps)  # Load atlas image

# Get AAL atlas labels and coordinates
labels = atlas.labels
atlas_data = atlas_img.get_fdata()

# Get AAL atlas spatial coordinates
coords = np.array(np.nonzero(atlas_data)).T  # Get coordinates of non-zero elements

# Get center coordinates for each region
unique_labels = np.unique(atlas_data[coords[:, 0], coords[:, 1], coords[:, 2]])
coords_unique = []
for label in unique_labels:
    # Get all coordinates for this label
    label_coords = coords[atlas_data[coords[:, 0], coords[:, 1], coords[:, 2]] == label]
    # Calculate mean coordinate
    mean_coord = label_coords.mean(axis=0)
    coords_unique.append(mean_coord)

# Convert coordinates to numpy array and ensure correct shape
coords = np.array(coords_unique)

# Convert to MNI space
affine = atlas_img.affine  # Get AAL atlas affine matrix
coords_mni = np.dot(coords, affine[:3, :3].T) + affine[:3, 3]  # Apply affine transformation

# Define brain lobe groupings
brain_regions = {
    "Frontal": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33],
    "Occipital": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    "Parietal": [34, 35, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
    "Subcortical": [28, 29, 70, 71, 72, 73, 74, 75, 76, 77],
    "Temporal": [36, 37, 38, 39, 40, 41, 54, 55, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
    "Cerebellum": [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
}

# Iterate through each brain lobe and plot connectivity
for region_name, region_indices in brain_regions.items():
    # Create connectivity matrix containing only specific brain lobe regions
    region_matrix = first_subject_matrix[np.ix_(region_indices, region_indices)]
    
    # Plot connectivity
    fig, ax = plt.subplots(figsize=(10, 10))
    plotting.plot_connectome(
        region_matrix, 
        coords_mni[region_indices],  # Use MNI coordinates for this lobe
        title=f'{region_name} connectivity network', 
        edge_threshold='95%',  # Can adjust this parameter
        colorbar=True  # Add color bar to show connection strength
    )
    
    # Display brain background
    display = plotting.plot_img(atlas_img, threshold=0, display_mode='ortho', cut_coords=(0, 0, 0), title=f'{region_name} Atlas', axes=ax)
    
    # Save figure
    plt.savefig(f'{region_name}_connectome_subject_1.png')
    plt.show()