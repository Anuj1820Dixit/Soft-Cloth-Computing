import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
import cv2

# Path to your dataset
dataset_path = '/kaggle/input/cloth-dataset'

# Image dimensions
img_width, img_height = 1200, 1600
channels = 3

def load_and_resize_images(dataset_path, target_size=(320, 416)):
    images = []
    for img_file in sorted(os.listdir(dataset_path)):
        if img_file.endswith(".jpg"):
            
            img_path = os.path.join(dataset_path, img_file)
            img = cv2.imread(img_path)  # Read the image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            images.append(img)
    images = np.array(images, dtype=np.float32)
    images = (images - 127.5) / 127.5
    return images

# Example usage
dataset_path = '/kaggle/input/cloth-dataset'
images = load_and_resize_images(dataset_path)
print(f"Loaded and resized {len(images)} images.")

print("Shape of loaded images:", images.shape)

from tensorflow.keras.layers import Dense, Dropout, Reshape,Conv2D, Flatten, Conv2DTranspose, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.models import Sequential, Model

def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128 * 26 * 20, activation="relu", input_dim=latent_dim))  # Starting from a smaller feature map
    model.add(Reshape((26, 20, 128)))  # Reshape to size close to the final dimensions by a factor of 2^n
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))  # Upscaling to 52x40
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))  # Upscaling to 104x80
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'))  # Upscaling to 208x160
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh'))  # Final upscaling to 416x320
    return model

generator = build_generator(100)

def build_discriminator(img_shape=(416, 320, 3)):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))  # Downscale
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))  # Further downscaling
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

latent_dim = 100
discriminator.trainable = False  # Important to disable training on discriminator when training the combined model
z = Input(shape=(latent_dim,))
img = generator(z)
valid = discriminator(img)
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer='adam')

# Generate a single image to test
test_noise = np.random.normal(0, 1, (1, 100))
test_image = generator.predict(test_noise)
print("Generator output shape:", test_image.shape)

# Check if the discriminator can process this image
test_validity = discriminator.predict(test_image)
print("Discriminator output shape:", test_validity.shape)

import matplotlib.pyplot as plt

def save_images(epoch, generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    print(f"Saving images at epoch {epoch}")
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescaling to [0, 1]
    
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i, :, :, :], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs('generated_images', exist_ok=True)
    plt.savefig(f'generated_images/epoch_{epoch}.png')
    plt.close()

def train(generator, discriminator, combined, images, epochs, batch_size, save_interval):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, images.shape[0], batch_size)
        imgs = images[idx]

        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = combined.train_on_batch(noise, valid)

        if epoch % save_interval == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
            save_images(epoch, generator)

# Example usage
train(generator, discriminator, combined, images, epochs=10000, batch_size=16, save_interval=100)

from IPython.display import Image, display
import os

image_path = '/kaggle/working/generated_images/epoch_100.png'

# Display the image
display(Image(filename=image_path))
