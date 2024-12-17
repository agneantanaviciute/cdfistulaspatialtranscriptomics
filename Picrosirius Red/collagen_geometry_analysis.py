# %%
import tifffile
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
import os
from skimage.io import imread
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import erosion, opening, disk, skeletonize, remove_small_objects, thin
from skimage.filters.rank import entropy
from skimage.measure import shannon_entropy, label, regionprops
from skimage.util import img_as_ubyte, img_as_bool
from skimage.restoration import denoise_bilateral
from skimage import exposure
from skimage.feature import canny, corner_peaks
from tqdm import tqdm
import matplotlib
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance_matrix
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.cluster import KMeans
import umap
import re
from skimage import io
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.morphology import skeletonize, opening, disk
from skimage.filters import threshold_otsu
from tqdm import tqdm
from pathlib import Path
from skimage.measure import perimeter, label, regionprops
import argparse

# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Process paths for tile processing.")
    
    # Adding the arguments
    parser.add_argument('--path_to_tiles', type=str, required=True, help='Path to the tiles directory')
    parser.add_argument('--path_to_jpeg', type=str, required=True, help='Path to the JPEG file')
    parser.add_argument('--path_to_outputs', type=str, required=True, help='Path to the outputs directory')

    return parser.parse_args()


# Parse command-line arguments
args = parse_args()
    
# Accessing the paths
path_to_tiles = args.path_to_tiles
path_to_jpeg = args.path_to_jpeg
path_to_outputs = args.path_to_outputs

print(f"Path to tiles: {path_to_tiles}")
print(f"Path to JPEG: {path_to_jpeg}")
print(f"Path to outputs: {path_to_outputs}")



# %%
full_image = io.imread(path_to_jpeg)

# %%
plt.figure(figsize=(8, 8))
plt.imshow(full_image) 
plt.title('Original Image')
plt.show()

# %%
red_channel = full_image[:, :, 0]
plt.figure(figsize=(8, 8))
plt.imshow(red_channel, cmap='gray') 
plt.title('Red Channel')
plt.axis('off')
plt.show()

# %%
white_threshold = 240 
non_white_mask = np.all(full_image < white_threshold, axis=-1)
non_white_red_values = red_channel[non_white_mask]

# %%

non_white_red_values = non_white_red_values / np.max(red_channel)

# %%

red_channel = red_channel / np.max(red_channel)

# %%
thresh = threshold_otsu(non_white_red_values)
thresh = 0.15


# %%
binary_red_channel = (red_channel > thresh).astype(np.uint8)

# %%
# Plot the binarized red channel
plt.figure(figsize=(8, 8))
plt.imshow(binary_red_channel, cmap='gray') 
plt.title('Binarized Red Channel')
plt.show()

# %%
binary_red_channel = (red_channel < thresh).astype(np.uint8) 

# %%
# Plot the binarized red channel
plt.figure(figsize=(8, 8))
plt.imshow(binary_red_channel, cmap='gray') 
plt.title('Binarized Red Channel')
plt.show()

# %%
plt.figure(figsize=(8, 8))
plt.imshow(binary_red_channel, cmap='gray') 
plt.title('Binarized Red Channel')
plt.savefig(os.path.join(path_to_outputs, "binarised_image.jpg"), dpi=300, bbox_inches='tight') 
plt.close() 


# %%
binary_red_channel

# %%
# Load images from folder and extract x, y coordinates
def load_images_and_coordinates_from_folder(folder):
    images = []
    coordinates = []
    # Regular expression to capture x and y coordinates
    coordinate_pattern = re.compile(r"x=(\d+),y=(\d+)")

    for filename in os.listdir(folder):
        if filename.endswith(".tif"):  
            # Read the image
            img = imread(os.path.join(folder, filename))
            if img is not None:
                # Normalize and convert the image to an array
                img = img_to_array(img) / 255.0  # Normalize to [0, 1]
                images.append(img)
                
                # Extract x, y coordinates using regex
                match = coordinate_pattern.search(filename)
                if match:
                    x = int(match.group(1))  # Extract x coordinate
                    y = int(match.group(2))  # Extract y coordinate
                    coordinates.append((x, y))
                else:
                    coordinates.append((None, None))  

    return np.array(images), coordinates


# %%
def binarize_red_channel(image):
    """Binarize the image based only on the red channel using Otsu's thresholding."""
    # Extract the red channel (assuming the image is in RGB format)
    red_channel = image[:, :, 0]
    binary_image = red_channel < thresh
    
    return binary_image

# %%
def preprocess_image(binary_image):
    """
    Preprocess the image after binarisation by applying erosion and opening to clean it.
    """
    # Step 1: Apply erosion to thin the fibers and remove noise
    eroded_image = erosion(binary_image, disk(1))
    
    # Step 2: Apply morphological opening to smooth fibers and remove small noise
    opened_image = opening(eroded_image, disk(1))
    
    return opened_image

# %%
def prune_skeleton(skeleton, min_size=30):
    """Prune small branches from the skeleton by removing small objects."""
    pruned_skeleton = remove_small_objects(skeleton, min_size=min_size)
    return pruned_skeleton

def preprocess_and_skeletonize_with_morphology(image):
    """
    Preprocess the image by skeletonizing.
    """
    # Skeletonize the processed image
    skeleton = skeletonize(image)
    pruned_skeleton = prune_skeleton(skeleton, min_size=2)
    return pruned_skeleton

# %%
def calculate_anisotropy_from_skeleton(skeleton):
    """
    Calculate the anisotropy of the skeletonized image by measuring the variance in fiber orientations.
    """
    # Apply Sobel filter to compute the gradients
    sobel_x = sobel(skeleton, axis=0)  # Gradient in x direction
    sobel_y = sobel(skeleton, axis=1)  # Gradient in y direction
    
    # Calculate the orientation of fibers
    orientations = np.arctan2(sobel_y, sobel_x) * (180.0 / np.pi)  # Convert to degrees
    
    # Compute the anisotropy as the variance of fiber orientations
    anisotropy = np.var(orientations)
    
    return anisotropy

# %%
def box_count(Z, k):
    """Count the number of non-empty boxes of size kxk in a binary image Z."""
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)
    
    return len(np.where(S > 0)[0])

def fractal_dimension(Z):
    """Calculate the fractal dimension of a 2D binary image Z using box-counting."""
    # Convert image to binary if not already
    Z = Z > 0

    # Minimal dimension of the image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to the size of the image
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Create a list of box sizes
    sizes = np.logspace(np.log10(1), np.log10(n), num=10, endpoint=True, base=2).astype(int)

    # Count the boxes for each size
    counts = [box_count(Z, size) for size in sizes]

    # Fit a line to log-log plot
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

    return -coeffs[0]  # The fractal dimension is the negative slope

# %%
def lacunarity(image, box_sizes):
    """
    Calculate the lacunarity of a binary image using different box sizes.
    Lacunarity is a measure of texture and spatial heterogeneity.
    """
    lacunarities = []
    for box_size in box_sizes:
        box_count = []
        # Slide a box of size `box_size` over the image
        for i in range(0, image.shape[0], box_size):
            for j in range(0, image.shape[1], box_size):
                box = image[i:i + box_size, j:j + box_size]
                # Count the number of white pixels in the box
                box_count.append(np.sum(box))
        # Calculate variance and mean of box counts
        box_count = np.array(box_count)
        mean = np.mean(box_count)
        variance = np.var(box_count)
        if mean > 0:
            lacunarity_value = variance / mean ** 2
        else:
            lacunarity_value = 0
        lacunarities.append(lacunarity_value)
    
    return lacunarities


# %%
def calculate_fiber_length(skeleton):
    """
    Calculate the total and individual fiber lengths from the skeletonized image.
    """
    # Label the skeleton
    labeled_skeleton, num_fibers = label(skeleton, return_num=True)
    
    # List to store the lengths of individual connected components
    fiber_lengths = []
    
    for region in regionprops(labeled_skeleton):
        # Length of each connected component (perimeter of the region in skeleton)
        fiber_length = region.perimeter
        fiber_lengths.append(fiber_length)
    
    fiber_lengths = np.array(fiber_lengths)
    
    # Calculate total length, mean, and quantiles
    total_length = fiber_lengths.sum()
    mean_length = np.mean(fiber_lengths)
    quantiles_length = np.quantile(fiber_lengths, [0.05, 0.5, 0.95])
    
    return total_length, mean_length, quantiles_length, fiber_lengths

def calculate_fiber_thickness(binary_image, skeleton):
    """
    Calculate the thickness of fibers by using a distance transform.
    """
    # Step 1: Apply distance transform on the binary image
    distance_map = distance_transform_edt(binary_image)
    
    # Step 2: Measure the thickness along the skeleton
    thickness_values = distance_map[skeleton > 0] * 2  # Multiply by 2 to get diameter
    
    # Calculate mean and quantiles for the thickness
    mean_thickness = np.mean(thickness_values)
    quantiles_thickness = np.quantile(thickness_values, [0.05, 0.5, 0.95])
    
    return mean_thickness, quantiles_thickness, thickness_values

def analyze_skeleton_connectivity(skeleton):
    """
    Analyze the connectivity of the skeleton (number of connected components).
    """
    labeled_skeleton, num_connected_components = label(skeleton, return_num=True)
    return num_connected_components

def analyze_fibers(image):
    """
    Full pipeline: Preprocess the image, skeletonize, and calculate fiber length and thickness.
    """
    # Step 1: Preprocess the image
    binary_image = binarize_red_channel(image)
    binary_image = preprocess_image(binary_image)
    # Step 2: Skeletonize the image to get the fiber skeletons
    skeleton = preprocess_and_skeletonize_with_morphology(binary_image)
    
    # Step 3: Calculate total and individual fiber lengths
    total_length, mean_length, quantiles_length, fiber_lengths = calculate_fiber_length(skeleton)
    
    # Step 4: Calculate fiber thickness
    mean_thickness, quantiles_thickness, thickness_values = calculate_fiber_thickness(binary_image, skeleton)
    
    # Step 5: Analyze skeleton connectivity
    num_connected_components = analyze_skeleton_connectivity(skeleton)
    
    return {
        'total_length': total_length,
        'mean_length': mean_length,
        'quantiles_length': quantiles_length,
        'fiber_lengths': fiber_lengths,
        'mean_thickness': mean_thickness,
        'quantiles_thickness': quantiles_thickness,
        'thickness_values': thickness_values,
        'num_connected_components': num_connected_components,
        'skeleton': skeleton
    }

# %%
def calculate_mean_free_path(skeleton):
    """
    Calculate the mean free path of fibers in a binary image.
    
    Args:
        binary_image (numpy array): Binary image where fibers are white (1) and background is black (0).
        
    Returns:
        mean_free_path (float): The mean free path between fiber endpoints or branch points.
    """
    try:
        # Detect endpoints or branch points
        labeled_skeleton, num_labels = label(skeleton, return_num=True)
        endpoints = np.argwhere(skeleton > 0)  # Get the coordinates of all points in the skeleton

        if len(endpoints) < 2:
            return -1  # Not enough points to calculate MFP
        
        # Calculate the distance between all endpoint pairs
        distances = distance_matrix(endpoints, endpoints)

        # Ignore distances between the same point (diagonal of the matrix)
        distances = distances[distances > 0]

        # Calculate the mean free path
        mean_free_path = np.mean(distances)
        
        return mean_free_path
    
    except Exception as e:
        print(f"Error calculating mean free path: {e}")
        return -1



# %%
def calculate_mean_free_path_of_empty_space(binary_image):
    """
    Calculate the mean free path of empty space in a binary image.
    
    Args:
        binary_image (numpy array): Binary image where fibers are white (1) and background is black (0).
        
    Returns:
        mean_free_path_empty_space (float): The mean free path of empty space.
    """
    try:
        # Invert the binary image so background (empty space) is 1 and fibers are 0
        inverted_image = np.logical_not(binary_image)

        # Calculate the distance transform of the empty space (background)
        distance_map = distance_transform_edt(inverted_image)
        
        # Extract distances for background pixels (inverted_image > 0)
        background_distances = distance_map[inverted_image > 0]
        
        # Calculate the mean of the distances (mean free path of empty space)
        mean_free_path_empty_space = np.mean(background_distances)
        
        return mean_free_path_empty_space
    
    except Exception as e:
        print(f"Error calculating mean free path of empty space: {e}")
        return -1



# %%
def calculate_perimeter_area_ratio(binary_image):
    """
    Calculate the perimeter-area ratio of fibers in a binary image.
    
    Args:
        binary_image (numpy array): Binary image where fibers are white (1) and background is black (0).
    
    Returns:
        perimeter_area_ratio (float): The ratio of perimeter to area for the fibers in the image.
    """
    try:
        # Label the connected components (fibers) in the binary image
        labeled_image = label(binary_image)
        
        # Calculate total perimeter and total area
        total_perimeter = 0
        total_area = 0
        
        for region in regionprops(labeled_image):
            # Add the perimeter and area of each connected component
            total_perimeter += perimeter(region.image)
            total_area += region.area
        
        # Calculate perimeter-area ratio
        if total_area > 0:
            perimeter_area_ratio = total_perimeter / total_area
        else:
            perimeter_area_ratio = -1  # Handle case where there's no fiber
        
        return perimeter_area_ratio
    
    except Exception as e:
        print(f"Error calculating perimeter-area ratio: {e}")
        return -1


# %%
def calculate_perimeter_area_ratio_mean(binary_image):
    """
    Calculate the perimeter-area ratio of fibers in a binary image.
    
    Args:
        binary_image (numpy array): Binary image where fibers are white (1) and background is black (0).
    
    Returns:
        perimeter_area_ratio (float): The ratio of perimeter to area for the fibers in the image.
    """
    try:
        # Label the connected components (fibers) in the binary image
        labeled_image = label(binary_image)
        
        # Calculate total perimeter and total area
        total = 0
        n = 0
        for region in regionprops(labeled_image):
            # Add the perimeter and area of each connected component
            total += perimeter(region.image) /region.area
            n = n +1
        
        # Calculate perimeter-area ratio
        perimeter_area_ratio = total / n  
        
        return perimeter_area_ratio
    
    except Exception as e:
        print(f"Error calculating perimeter-area ratio: {e}")
        return -1


# %%
def calculate_mean_eccentricity(binary_image):
    """
    Calculate the eccentricity of fibers in a binary image.
    
    Args:
        binary_image (numpy array): Binary image where fibers are white (1) and background is black (0).
        
    Returns:
        eccentricities (list): List of eccentricities for each connected fiber in the image.
    """
    try:
        # Label the connected components (fibers) in the binary image
        labeled_image = label(binary_image)
        
        # List to store eccentricities of each fiber
        eccentricities = 0
        n = 0
        for region in regionprops(labeled_image):
            # Extract eccentricity of each region (fiber)
            eccentricities += region.eccentricity
            n = n +1
        
        mean = eccentricities / n
        return mean
    
    except Exception as e:
        print(f"Error calculating eccentricity: {e}")
        return []


# %%
def calculate_average_elongation(binary_image):
    """
    Calculate the average elongation of fibers in a binary image.
    
    Args:
        binary_image (numpy array): Binary image where fibers are white (1) and background is black (0).
        
    Returns:
        average_elongation (float): The average elongation of all fibers in the image.
    """
    try:
        # Label the connected components (fibers) in the binary image
        labeled_image = label(binary_image)
        
        # List to store elongation of each fiber
        elongations = []
        
        for region in regionprops(labeled_image):
            # Extract the major and minor axis lengths
            major_axis_length = region.major_axis_length
            minor_axis_length = region.minor_axis_length
            
            # Calculate elongation (a / b)
            if minor_axis_length > 0:
                elongation = major_axis_length / minor_axis_length
            else:
                elongation = 0  # Handle case where the minor axis is zero (e.g., for line-like objects)
            
            elongations.append(elongation)
        
        # Calculate the average elongation
        if elongations:
            average_elongation = np.mean(elongations)
        else:
            average_elongation = -1  # Handle case with no valid fibers
        
        return average_elongation
    
    except Exception as e:
        print(f"Error calculating average elongation: {e}")
        return -1


# %%
def calculate_mean_entropy(binary_image, patch_size=None):
    """
    Calculate the mean entropy of the entire binary image or of patches of the image.
    
    Args:
        binary_image (numpy array): Binary image where fibers are white (1) and background is black (0).
        patch_size (tuple, optional): Size of the patches to divide the image into (height, width).
                                      If None, calculate entropy over the entire image.
        
    Returns:
        mean_entropy (float): The mean entropy of the image or its patches.
    """
    try:
        # If patch_size is None, calculate entropy of the entire image
        if patch_size is None:
            return shannon_entropy(binary_image)
        
        # Split the image into patches of the specified size
        height, width = binary_image.shape
        patch_height, patch_width = patch_size
        
        entropies = []
        
        # Iterate over patches
        for i in range(0, height, patch_height):
            for j in range(0, width, patch_width):
                patch = binary_image[i:i + patch_height, j:j + patch_width]
                
                # Calculate entropy for each patch
                entropy = shannon_entropy(patch)
                entropies.append(entropy)
        
        # Calculate mean entropy
        mean_entropy = np.mean(entropies)
        return mean_entropy
    
    except Exception as e:
        print(f"Error calculating mean entropy: {e}")
        return -1

# %%
def calculate_collagen_fraction_red_channel(image):
    """
    Calculate the collagen fraction of an image by thresholding the red channel.
    This is based on Picrosirius Red staining, where collagen appears prominently in the red channel.
    """
    # Calculate collagen fraction as the proportion of collagen-rich pixels in the red channel
    collagen_fraction = np.sum(image) / np.size(image)
    
    return collagen_fraction
# %%
image_patches, image_coordinates = load_images_and_coordinates_from_folder(path_to_tiles)
# Print example results
print("First image shape:", image_patches[0].shape)
print("First image coordinates:", image_coordinates[0])

# %%
def process_image_patches_with_analysis(image_patches, image_coordinates, output_directory):
    """
    Loop over image patches, binarize, preprocess, skeletonize, and apply all image analysis functions.
    Store the results along with image coordinates in a DataFrame, and write to a CSV.

    Args:
        image_patches (list of numpy arrays): List of image patch arrays.
        image_coordinates (list of tuple): List of (x, y) coordinates corresponding to image patches.
        output_directory (str): Directory where the CSV file will be saved.
    
    Returns:
        DataFrame with all calculated metrics for each image patch.
    """
    
    # Initialize list to store results
    results = []

    # Loop over all image patches with a progress bar
    for idx, image in enumerate(tqdm(image_patches, desc="Processing Image Patches")):
        try:

            # Binarize the red channel using Otsu's threshold - previously calculated from whole image
            binary_image = binarize_red_channel(image)

            # Preprocess the binary image
            binary_image = preprocess_image(binary_image)

            # Skeletonize the image
            pruned_skeleton = preprocess_and_skeletonize_with_morphology(binary_image)

            # Perform all calculations from the notebook
            entropy_value = calculate_mean_entropy(binary_image)
            fractal_dim = fractal_dimension(binary_image)
            lacunarity_value = lacunarity(binary_image, box_sizes=[4, 8, 16])
            total_fiber_length, mean_fiber_length, quantiles_fiber_length, _ = calculate_fiber_length(pruned_skeleton)
            mean_fiber_thickness, quantiles_thickness, _ = calculate_fiber_thickness(binary_image, pruned_skeleton)
            num_connected_components = analyze_skeleton_connectivity(pruned_skeleton)
            anisotropy = calculate_anisotropy_from_skeleton(pruned_skeleton)
            mean_free_path = calculate_mean_free_path(pruned_skeleton)
            mean_free_path_empty_space = calculate_mean_free_path_of_empty_space(binary_image)
            perimeter_area_ratio = calculate_perimeter_area_ratio(binary_image)
            mean_eccentricity = calculate_mean_eccentricity(binary_image)
            average_elongation = calculate_average_elongation(binary_image)
            fraction_collagen = calculate_collagen_fraction_red_channel(binary_image)
            # Store the results in a dictionary
            result = {
                'x_coordinate': image_coordinates[idx][0],
                'y_coordinate': image_coordinates[idx][1],
                'entropy': entropy_value,
                'fractal_dimension': fractal_dim,
                'lacunarity_4': lacunarity_value[0],
                'lacunarity_8': lacunarity_value[1],
                'lacunarity_16': lacunarity_value[2],
                'total_fiber_length': total_fiber_length,
                'mean_fiber_length': mean_fiber_length,
                'fiber_length_05': quantiles_fiber_length[0],
                'fiber_length_50': quantiles_fiber_length[1],
                'fiber_length_95': quantiles_fiber_length[2],
                'mean_fiber_thickness': mean_fiber_thickness,
                'fiber_thickness_05': quantiles_thickness[0],
                'fiber_thickness_50': quantiles_thickness[1],
                'fiber_thickness_95': quantiles_thickness[2],
                'num_connected_components': num_connected_components,
                'anisotropy': anisotropy,
                'mean_free_path': mean_free_path,
                'mean_free_path_empty_space': mean_free_path_empty_space,
                'perimeter_area_ratio': perimeter_area_ratio,
                'mean_eccentricity': mean_eccentricity,
                'average_elongation': average_elongation,
                'fraction_collagen' : fraction_collagen
                
            }
            
            # Append result to list
            results.append(result)
        
        except Exception as e:
            print(f"Error processing image patch {idx}: {e}")
    
    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    
    # Write the DataFrame to CSV
    output_path = Path(output_directory) / "image_patch_analysis.csv"
    df.to_csv(output_path, index=False)
    
    return df


# %%
df_results = process_image_patches_with_analysis(image_patches, image_coordinates, path_to_outputs)

# %%
def plot_metrics_by_coordinates(results_df, output_folder):
    """
    Plots all the metrics in the DataFrame against the patch coordinates.
    
    Args:
    - results_df (pd.DataFrame): The DataFrame containing the metrics and coordinates (x, y).
    - output_folder (str): Path to the folder where the plots will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract x and y coordinates from the DataFrame
    x_coords = results_df['x_coordinate']
    y_coords = results_df['y_coordinate']
    
    # Loop over all columns (except x and y) to plot each metric
    for column in results_df.columns:
        if column not in ['x_coordinate', 'y_coordinate']:  # Skip the coordinate columns
                
                # Create a scatter plot of the metric in the patch coordinate system
                plt.figure(figsize=(8, 8))
                plt.scatter(x_coords, y_coords, c=results_df[column], cmap='viridis', s=30, edgecolor='k')
                plt.colorbar(label=column)
                plt.title(f'{column} by Patch Coordinates')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.grid(True)
                
                # Save the plot to file
                plot_filename = os.path.join(output_folder, f'{column}_by_coordinates.png')
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to avoid memory overload

# %%
df_results

# %%
plot_metrics_by_coordinates(df_results, path_to_outputs)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler

def plot_heatmap_with_clustering(df_results, output_file, z_threshold=3):
    # Remove columns with zero variance
    # Apply Z-score filtering to remove rows with extreme outliers
    df_filtered = df_results[(np.abs(df_results.apply(lambda x: (x - x.mean()) / x.std())) < z_threshold).all(axis=1)]

    # After filtering, remove columns with zero variance
    df_no_zero_variance = df_filtered.loc[:, df_filtered.var() != 0]

    # Standardize the data
    scaler = StandardScaler()
    df_metrics_scaled = pd.DataFrame(scaler.fit_transform(df_no_zero_variance), 
                                     index=df_no_zero_variance.index, 
                                     columns=df_no_zero_variance.columns)

    # Perform hierarchical clustering
    row_linkage = linkage(df_metrics_scaled, method='ward')

    # Create a clustered heatmap
    sns.clustermap(df_metrics_scaled, row_cluster=True, col_cluster=True, 
                   row_linkage=row_linkage, cmap='vlag', figsize=(10, 10), 
                   standard_scale=1)  # standard_scale=1 normalizes the data within the heatmap

    # Save the heatmap to the specified file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# Example usage
plot_heatmap_with_clustering(df_results, os.path.join(path_to_outputs,'heatmap_filtered.png'), z_threshold=3)


# %%
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def plot_heatmap_with_clustering(df_results, output_file, z_threshold=3, num_clusters=4):
    df_filtered = df_results[(np.abs(df_results.apply(lambda x: (x - x.mean()) / x.std())) < z_threshold).all(axis=1)]

    # After filtering, remove columns with zero variance
    df_no_zero_variance = df_filtered.loc[:, df_filtered.var() != 0]

    # Standardize the data
    scaler = StandardScaler()
    df_metrics_scaled = pd.DataFrame(scaler.fit_transform(df_no_zero_variance), 
                                     index=df_no_zero_variance.index, 
                                     columns=df_no_zero_variance.columns)

    # Perform hierarchical clustering
    row_linkage = linkage(df_metrics_scaled, method='ward')

    # Create a clustered heatmap
    sns.clustermap(df_metrics_scaled, row_cluster=True, col_cluster=True, 
                   row_linkage=row_linkage, cmap='vlag', figsize=(10, 10), 
                   standard_scale=1)  # standard_scale=1 normalizes the data within the heatmap

    # Save the heatmap to the specified file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Cutting the tree to form clusters
    clusters = fcluster(row_linkage, num_clusters, criterion='maxclust')

    # Adding the cluster labels to the filtered table (df_filtered)
    df_filtered['Cluster'] = clusters

    # Now merge the cluster labels back to the original dataframe based on the index
    df_results_with_clusters = df_results.loc[df_filtered.index].copy()
    df_results_with_clusters['Cluster'] = clusters

    # Plot the clusters over the x and y coordinates
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='x_coordinate', y='y_coordinate', hue='Cluster', palette='Set2', data=df_results_with_clusters, s=30)
    plt.title('Clusters plotted over x_coordinate and y_coordinate')
    plt.xlabel('x_coordinate')
    plt.ylabel('y_coordinate')
    
    # Save the plot
    plt.savefig(output_file.replace('.png', '_clusters.png'), dpi=300)
    plt.close()

# Example usage
plot_heatmap_with_clustering(df_results, os.path.join(path_to_outputs,'heatmap_filtered.png'), z_threshold=3, num_clusters=6)



