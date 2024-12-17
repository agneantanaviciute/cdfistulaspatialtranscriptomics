import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from tensorflow.keras.utils import Sequence
from skimage import exposure
# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Process paths for tile processing.")
    
    # Adding the arguments
    parser.add_argument('--path_to_tiles', type=str, required=True, help='Path to the tiles directory')
    parser.add_argument('--path_to_outputs', type=str, required=True, help='Path to the outputs directory')

    return parser.parse_args()


# Parse command-line arguments
args = parse_args()
    
# Accessing the paths
path_to_tiles = args.path_to_tiles
path_to_outputs = args.path_to_outputs

print(f"Path to tiles: {path_to_tiles}")
print(f"Path to outputs: {path_to_outputs}")

# Sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

from tensorflow.keras.layers import Layer

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


cae = load_model("vae_vgg_model.h5",  custom_objects={'Sampling': Sampling})
encoder = cae.get_layer('encoder')

def extract_latent_features(encoder, image_dir, output_csv, target_size=(128, 128), batch_size=32):
    """
    Extracts latent dimensions (z_mean) for each image patch using the provided encoder and saves the results in a CSV file.

    Args:
    - encoder: The loaded encoder model.
    - image_dir: Path to the directory containing image patches.
    - output_csv: Path to save the CSV file with latent dimensions.
    - target_size: Tuple specifying the target size for resizing images.
    - batch_size: Number of images to process in each batch.
    """
    data = []

    # Collect all image file paths
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')]
    num_images = len(image_paths)
    num_batches = (num_images + batch_size - 1) // batch_size

    # Iterate over batches
    for batch_index in range(num_batches):
        batch_paths = image_paths[batch_index * batch_size:(batch_index + 1) * batch_size]
        batch_images = []
        batch_info = []

        # Load and preprocess images in the batch
        for image_path in batch_paths:
            image_file = os.path.basename(image_path)

            # Extract x and y coordinates from the image file name
            try:
                x_coord = int(image_file.split('x=')[1].split(',')[0])
                y_coord = int(image_file.split('y=')[1].split(',')[0])
            except (IndexError, ValueError):
                print(f"Skipping {image_file} due to naming issue.")
                continue

            # Load and preprocess the image
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
            img_array = exposure.adjust_gamma(img_array, gamma=0.3)
            batch_images.append(img_array)
            batch_info.append((x_coord, y_coord))

        # Convert batch_images to a numpy array and predict latent dimensions
        if batch_images:
            batch_images = np.array(batch_images)
            z_mean, _, _ = encoder.predict(batch_images)

            # Store the results
            for (x_coord, y_coord), latent_vector in zip(batch_info, z_mean):
                row = [x_coord, y_coord] + latent_vector.tolist()
                data.append(row)

    # Create a DataFrame and save it to CSV
    columns = ['x_coordinate', 'y_coordinate'] + [f'dim{i+1}' for i in range(len(z_mean[0]))]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Latent dimensions saved to {output_csv}")

# Example usage
extract_latent_features(encoder, path_to_tiles,
                        os.path.join(path_to_outputs,'vae_vgg_model10_latent_features.csv'),
                        target_size=(128, 128),
                        batch_size=128)