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
