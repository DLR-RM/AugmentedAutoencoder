
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
# import seaborn as sns
# Change figure aesthetics
# sns.set_context('talk', font_scale=1.2, rc={'lines.linewidth': 1.5})


# Load dataset
dataset_zip = np.load('/home_local/sund_ma/data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

print('Keys in the dataset:', dataset_zip.keys())
imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
latents_classes = dataset_zip['latents_classes']
metadata = dataset_zip['metadata'][()]
print('Metadata: \n', metadata)
print(latents_classes) 

# Define number of values per latents and functions to convert to indices
latents_sizes = metadata['latents_sizes']
latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))

print(latents_bases)
def latent_to_index(latents):
  return np.dot(latents, latents_bases).astype(int)


def sample_latent(size=1):
  samples = np.zeros((size, latents_sizes.size))
  for lat_i, lat_size in enumerate(latents_sizes):
    samples[:, lat_i] = np.random.randint(lat_size, size=size)

  return samples


# Helper function to show images
def show_images_grid(imgs_, num_images=25):
  ncols = int(np.ceil(num_images**0.5))
  nrows = int(np.ceil(num_images / ncols))
  _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
  axes = axes.flatten()

  for ax_i, ax in enumerate(axes):
    if ax_i < num_images:
      ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
      ax.set_xticks([])
      ax.set_yticks([])
    else:
      ax.axis('off')

def show_density(imgs):
  _, ax = plt.subplots()
  ax.imshow(imgs.mean(axis=0), interpolation='nearest', cmap='Greys_r')
  ax.grid('off')
  ax.set_xticks([])
  ax.set_yticks([])

## Fix posX latent to left
# latents_sampled = sample_latent(size=5000)

latents_classes_heart = latents_classes[-245760:]
latents_classes_heart_rot = latents_classes_heart.copy()

latents_classes_heart_rot[:, 0] = 0
latents_classes_heart_rot[:, 1] = 2
latents_classes_heart_rot[:, 2] = 5
latents_classes_heart_rot[:, 4] = 16
latents_classes_heart_rot[:, 5] = 16
indices_sampled = latent_to_index(latents_classes_heart_rot)
imgs_sampled_rot = imgs[indices_sampled]
indices_sampled = latent_to_index(latents_classes_heart)
imgs_sampled_all = imgs[indices_sampled]

import cv2

for img in imgs_sampled_rot:
  rot_angle= np.random.rand()*360
  cent = int(img.shape[0]/2)
  M = cv2.getRotationMatrix2D((cent,cent),rot_angle,1)
  rot_img = cv2.warpAffine(img,M,(img.shape[0],img.shape[1]))
  print (rot_img.shape)
  if len(rot_img.shape)<3:
      rot_img=rot_img[:,:,np.newaxis]
  cv2.imshow('rot_img',rot_img*255)
  cv2.waitKey(0)

# Samples
print(imgs_sampled_rot.shape)
random_idcs = np.random.choice(len(imgs_sampled_rot),9)
heart_embed = imgs_sampled_rot[::latents_bases[3]]
print(heart_embed.shape)
show_images_grid(heart_embed[40:80], 40)
show_images_grid(imgs_sampled_all[random_idcs], 9)
plt.show()

# Show the density too to check
show_density(imgs_sampled)

## Fix orientation to 0.8 rad
latents_sampled = sample_latent(size=5000)
latents_sampled[:, 3] = 5
indices_sampled = latent_to_index(latents_sampled)
imgs_sampled = imgs[indices_sampled]

# Samples
show_images_grid(imgs_sampled, 9)

# Density should not be different than for all orientations
show_density(imgs_sampled)
