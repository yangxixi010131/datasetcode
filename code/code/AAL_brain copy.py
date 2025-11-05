# Load data
correlation_matrices = np.load('all_correlation_matrices103.npy')
first_subject_matrix = correlation_matrices[0]

# Load AAL atlas
atlas = datasets.fetch_atlas_aal()
atlas_img = image.load_img(atlas.maps)

# Calculate MNI coordinates
atlas_data = atlas_img.get_fdata()
coords = []
for label in np.unique(atlas_data):
    if label == 0:  # Skip background
        continue
    mask = atlas_data == label
    coords.append(np.array(np.where(mask)).mean(axis=1))
coords = np.array(coords)

# Transform to MNI space
affine = atlas_img.affine
coords_mni = np.dot(coords, affine[:3, :3].T) + affine[:3, 3]

# Define brain lobe regions
brain_regions = {
    "Frontal": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33],
    "Occipital": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    "Parietal": [34, 35, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
    "Subcortical": [28, 29, 70, 71, 72, 73, 74, 75, 76, 77],
    "Temporal": [36, 37, 38, 39, 40, 41, 54, 55, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
    "Cerebellum": [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
}

# Assign colors to each brain lobe
region_colors = {
    "Frontal": "#FF5733",        # Red
    "Occipital": "#33FF57",      # Green
    "Parietal": "#3357FF",       # Blue
    "Subcortical": "#FF33F6",    # Purple
    "Temporal": "#33FFF6",       # Cyan
    "Cerebellum": "#F6FF33"      # Yellow
}

# Node colors
node_colors = []
for region, indices in brain_regions.items():
    color = region_colors[region]
    node_colors.extend([color] * len(indices))

# Visualization settings
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16})
fig = plt.figure(figsize=(12, 10))

# Plot connectome
plotting.plot_connectome(
    adjacency_matrix=first_subject_matrix,
    node_coords=coords_mni,
    edge_threshold="93%",
    edge_cmap='YlOrRd',  # Use default colormap
    node_color=node_colors,
    node_size=50,
    display_mode='x',  # Show only mid-horizontal slice (left-right brain)
    figure=fig,
    edge_kwargs={'alpha': 0.6}
)

# Add legend
legend_ax = fig.add_axes([0.7, 0.1, 0.2, 0.1])
for region, color in region_colors.items():
    legend_ax.plot([], [], 'o', color=color, label=region)
legend_ax.legend(loc='center', frameon=False)
legend_ax.axis('off')

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig('brain_connectome.png', dpi=300, bbox_inches='tight')
plt.show()