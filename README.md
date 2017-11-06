# Tensorflow AutoEncoder

# Usage
### Preparatory Steps
*1. Create Workspace*
```bash
WS=/home_local/$USER/autoencoder_ws
mkdir $WS
```

*2. Init Workspace*
```bash
cd $WS
ae_init_workspace
```

*3. Source environment*
```bash
source /home_local/$USER/autoencoder_ws/setup.bash
```
This will set the variable AE_WORKSPACE_PATH

### Train A Model
*1. Copy the template config file*
```bash
cp $AE_WORKSPACE_PATH/cfg/template.cfg $AE_WORKSPACE_PATH/cfg/my_autoencoder.cfg
```
*2. Modify the config file*
```bash
gedit $AE_WORKSPACE_PATH/cfg/my_autoencoder.cfg
```

*3. Check that data is generated correctly*
(Press *ESC* to clos the window.)
```bash
ae_train my_autoencoder -d
```
This command does not start training.
Output:
![](docs/example_batch.png)

*4. Train the model*
```bash
ae_train my_autoencoder
```

*5. Create the embedding*
```bash
ae_embed my_autoencoder
```

### Use A Model
```python
import tensorflow as tf
from ae import factory

experiment_name = 'my_autoencoder'

codebook = factory.build_codebook_from_name(experiment_name)

with tf.Session() as sess:
	factory.restore_checkpoint(sess, tf.train.Saver(), experiment_name)
	img = webcam.snapshot()
	R = codebook.nearest_rotation(session, img)
	print R
```
Example output:
```bash
[[ 1.          0.          0.        ]
 [ 0.         -0.85065081 -0.52573111]
 [ 0.          0.52573111 -0.85065081]]
```
# Installation

```bash
make
```

# Config file parameters
```yaml
[Paths]
# Path to the model file. All formats supported by assimp should work. Tested with ply files.
MODEL_PATH: /net/rmc-lx0050/home_local/shared/ikea_mug_reduced.ply
# Path to some background image folder. Should contain a * as a placeholder for the image name.
BACKGROUND_IMAGES_GLOB: /net/rmc-lx0050/home_local/sund_ma/data/VOCdevkit/VOC2012/JPEGImages/*.jpg

[Dataset]
# Hight of the AE input layer
H: 128 
# Width of the AE input layer
W: 128 
# Distance from Camera to the object in mm
RADIUS: 650 
# Dimensions of the renderered image, it will be cropped and rescaled to H, W later.
RENDER_DIMS: (1200, 900) 
# Camera matrix used for rendering
K: [1000, 0, 500, 0, 1000, 500, 0, 0, 1] 
# Vertex scale. Vertices need to be scaled to millimeter
VERTEX_SCALE: 1000 
# Antialiasing factor used for rendering
ANTIALIASING: 16 
# This factor adds a border to the cropped image.
CROP_FACTOR: 1.2 
# Near plane
CLIP_NEAR: 10 
# Far plane
CLIP_FAR: 10000 

[Augmentation]
# During training an offset is sampled from Normal(0, CROP_OFFSET_SIGMA) and added to the ground truth crop.
CROP_OFFSET_SIGMA: 30 
# Code for the augmentations. Documentation: https://github.com/aleju/imgaug.
CODE: Sequential([ 
    Sometimes(0.5, Add((-20, 20), per_channel=0.3)),
    Sometimes(0.3, Invert(0.20, per_channel=True)),
    Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
    Sometimes(0.5, Multiply((0.6, 1.4))),
    Sometimes(0.5, ContrastNormalization((0.4, 2.3), per_channel=0.3)),
    #Sometimes(0.2, CoarseDropout( p=0.3, size_percent=0.01) )
], random_order=True)

[Embedding]
# minimum number of views from an evenly sampled viewsphere used to create the codebook
MIN_N_VIEWS: 500 
# number of inplane rotations used to create the codebook
NUM_CYCLO: 36 

[Network]
# Size of the latent space, for image size 128 this is the optimal value.
LATENT_SPACE_SIZE: 64 
# Number of filters used for convolution
NUM_FILTER: [32, 32, 64, 64] 
# Strides used for the convolution. Note: Stride 2 Convolutions are used instead of pooling layers.
STRIDES: [1, 2, 1, 2] 
KERNEL_SIZE_ENCODER: 5 
KERNEL_SIZE_DECODER: 5

[Training]
# Any other optimizer supported by tensorflow can be used here
OPTIMIZER: Adam 
# Train the network this number of steps
NUM_ITER: 20000
# Batchsize used for training
BATCH_SIZE: 64 
# Learning rate
LEARNING_RATE: 1e-4 
# Creates a checkpoint at after SAVE_INTERVAL steps.
SAVE_INTERVAL: 5000 

[Queue]
# Number of threads used for augmentation
NUM_THREADS: 10
# This is the queue size of the queue which is used for the training process.
QUEUE_SIZE: 10 
```

