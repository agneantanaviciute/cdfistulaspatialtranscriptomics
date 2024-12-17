import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
import re
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random
from tensorflow.keras.utils import Sequence
from tqdm.notebook import tqdm
import pandas as pd
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import os
import re
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage import exposure

class ImageTileGenerator(Sequence):
    def __init__(self, image_dir, batch_size, image_size=(512, 512), subset_size=None, shuffle=True):
        """
        Args:
        - image_dir: Directory containing image tiles in subdirectories.
        - batch_size: Number of images per batch.
        - image_size: Target size to resize images.
        - subset_size: The number of images to sample for training (subset of total images).
        - shuffle: Whether to shuffle the images at the start of each epoch.
        """
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle

        # Collect all image file paths from subdirectories
        self.image_list = []
        for subdir, _, files in os.walk(image_dir):
            self.image_list.extend([os.path.join(subdir, f) for f in files])

        self.subset_size = subset_size if subset_size is not None else len(self.image_list)
        self.on_epoch_end()

    def __len__(self):
        # Calculate the number of batches per epoch based on the subset size
        return int(np.floor(self.subset_size / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes for the current batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_files = [self.current_subset[i] for i in batch_indexes]

        # Generate data for the batch
        X, weights = self.__data_generation(batch_image_files)

        return X, X, weights  # Return inputs, targets (same for autoencoder), and weights

    def on_epoch_end(self):
        """Called at the end of each epoch to reshuffle and select a new subset of images."""
        # Reshuffle the image list if necessary
        self.indexes = np.arange(len(self.image_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

        # Select a new subset of images for this epoch
        subset_indexes = self.indexes[:self.subset_size]
        self.current_subset = [self.image_list[i] for i in subset_indexes]

        # Create indexes for the batches within the subset
        self.indexes = np.arange(self.subset_size)

    def __data_generation(self, batch_image_files):
        """Loads and processes a batch of images."""
        # Initialize the batch of images and weights
        batch_images = np.empty((self.batch_size, *self.image_size, 3))  # Assuming RGB images
        weights = np.empty((self.batch_size, 1))  # Weights for each image

        # Load and preprocess images
        for i, image_path in enumerate(batch_image_files):
            try:
                img = load_img(image_path, target_size=self.image_size)  # Load and resize image
                img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
                img_array = exposure.adjust_gamma(img_array, gamma=0.3)
                batch_images[i] = img_array

                # Compute the Shannon entropy of the image
                grayscale_img = rgb2gray(img_array)
                mean_entropy = shannon_entropy(grayscale_img)

                # Define a weight based on entropy (e.g., 1 for high entropy, <1 for low entropy)
                weights[i] = 1 if mean_entropy > 4 else 0.1  # Adjust the threshold as needed

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                batch_images[i] = np.zeros((self.image_size[0], self.image_size[1], 3))
                weights[i] = 0  # Assign zero weight for problematic images

        return batch_images, weights


# %%
# architecture
input_shape = (128, 128, 3)
latent_dim = 64

# Encoder
vgg_encoder = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
x = layers.Flatten()(vgg_encoder.output)
x = layers.Dense(256, activation='relu')(x)

# Freeze VGG16 layers
for layer in vgg_encoder.layers:
    layer.trainable = False

# Latent space
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# Sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Decoder
decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 256, activation='relu')(decoder_input)
x = layers.Reshape((8, 8, 256))(x)
x = layers.Conv2DTranspose(256, (3, 3), padding='same', activation='relu')(x)
x = layers.UpSampling2D(size=(2, 2))(x)  # 16x16x256
x = layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
x = layers.UpSampling2D(size=(2, 2))(x)  # 32x32x128
x = layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
x = layers.UpSampling2D(size=(2, 2))(x)  # 64x64x64
x = layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
x = layers.UpSampling2D(size=(2, 2))(x)  # 128x128x32
decoder_output = layers.Conv2DTranspose(3, (3, 3), padding='same', activation='sigmoid')(x)

# Define the model
encoder = models.Model(vgg_encoder.input, [z_mean, z_log_var, z], name='encoder')
decoder = models.Model(decoder_input, decoder_output, name='decoder')
vae_output = decoder(encoder(vgg_encoder.input)[2])
vae = models.Model(vgg_encoder.input, vae_output, name='vae')

vae.compile(optimizer='adam', loss='mse', sample_weight_mode="sample-wise")

# image generator
image_dir = '/final_brightfield_tiles'
batch_size = 512
image_size = (128, 128)
subset_size = 300000
generator = ImageTileGenerator(image_dir=image_dir, batch_size=batch_size, image_size=image_size, subset_size = subset_size)

# Train 
vae.fit(generator, epochs=30)

for layer in vgg_encoder.layers:
    layer.trainable = True

vae.fit(generator, epochs=20)

vae.save('model.h5')


