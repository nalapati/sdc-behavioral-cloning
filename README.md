# Using Deep Learning to Clone Driving Behavior

**NOTE: This project builds on work done in https://medium.com/udacity/teaching-a-machine-to-steer-a-car-d73217f2492c, specifically Ilya Edrenkin's approach of using a 3D Convolutional Neural Network.**

## Step 0: Infrastructure Setup
As part of Udacity challenge 2 in the blog above we got to a really good infrastructure setup on https://github.com/emef/sdc. This submission is a cut down version of the sdc project.

## Step 1: Model design
### Requirements:
1. The problem being solved here is given the center camera image from a car, predict the steering angle (fixed throttle). 
2. This problem does not require scale or rotation or translation invariance.
3. This problem requires generalization to different roads and lighting conditions.
4. Requires recovering from incorrect predictions to stay on the road and not go off track.

In view of the above, this solution implements a model that predicts the steering angle in radians given an input image
from the center camera of a car. It is sufficient to go shallow and not use a lot of non-linearity by using just a few
convolutional layers, but going wide by collecting a LOT of data and using a large number of filters in a few layers.
The 3D convolutional layers in the model allow learning time series information across different frames basically
the depth dimension amounts to timesteps or camera images from previous points in time.

### The model architecture:
**NOTE: only the lower half of the image is used in the model, and the input to the model is the current image and 9 previous images. Convolutions are done across time to extract temporal information in the driving data.**
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-behavioral-cloning/master/images/model.jpg)

## Step 2: Training on the sample data provided
The first step once the infrastructure was setup was to train the model on the sample data to determine if I could make
some predictions (any predictions), and also set it up so I had the basic metrics flowing through and models working 
end to end and used to drive the car.

## Step 3: Collecting data
Once the pipeline was setup end to end the approach was to start collecting data. I drove the car two and fro twice on
track 1 for the first dataset. (used a keyboard, no steering wheel or joystick, I expect a lot better results if I had smoother data,
I collected steering data by not using long presses on the keys on the sims but short bursts of key presses to better capture
the angles, this could be the reason for some of the understeering in the sharp turns in track 1 in the result).

## Step 4: Overfitting the collected data to solve Track 1
I tuned the model (number of layers, convolutional filters, filter sizes, hidden layer sizes) to come to a solution
that solved the track 1 test set, but the model overfit(later fixed to add dropouts + l2 regularization) in that it
would sway out after 2 laps, the data didn't have any correction samples. 

## Step 5: Collecting additional data
This was largely manual, but surprisingly required a lot of discipline to manage these datasets, I added increments to
a core dataset (each with its READMEs to track additions), My final dataset has the following composition:
* Track 1 to and fro 9 times.

![alt tag](https://raw.githubusercontent.com/nalapati/sdc-behavioral-cloning/master/images/center_2017_01_18_12_56_26_395.jpg)
* Track 1 correction to and fro 9 times.

![alt tag](https://raw.githubusercontent.com/nalapati/sdc-behavioral-cloning/master/images/center_2017_01_19_07_23_58_402.jpg)

![alt tag](https://raw.githubusercontent.com/nalapati/sdc-behavioral-cloning/master/images/center_2017_01_19_07_23_59_149.jpg)
* Track 1 sharp turns to and fro 12 times.

![alt tag](https://raw.githubusercontent.com/nalapati/sdc-behavioral-cloning/master/images/center_2017_01_19_09_40_52_539.jpg)
* Track 1 data on the bridge section.

![alt tag](https://raw.githubusercontent.com/nalapati/sdc-behavioral-cloning/master/images/center_2017_01_19_09_40_46_616.jpg)

The component pieces of data were collected incrementally to fix defects in the algorithm, the sharp turn datacollection run was to fix understeering and dataset imbalance. The bridge section is a wide road even wider than images from track 2 surprisingly, so the car tends to drift to the right (since I only use the bottom half of the image) and then it corrects because of the correction data incorporated into the dataset.

## Step 6 Model Training
In order to train the model I used the Dataset in the dataset.py class to load images images using a generator. The dataset was prepared offline to be split into train/test/validation in a 75%, 10% and 15% split respectively. The Dataset class in dataset.py uses an InfiniteImageLoadingGenerator that generates timestepped images. The images that are loaded are normalized in the load_image method.

BatchNormalization was used between convolutional layers to make convergence faster, it essentially reduces noise between layers by making the data well formed.

The number of epochs was chosen manually guarded by the EarlyStopping keras callback that stops the training if the validation rmse has not improved in 12 epochs, there was a bug in the initial version of the code that sets the delta change to 0, changed it to 0.0004. around the point where the training stops minimizing the training loss and validation error was converged(Updated the code to include the model checkpointer to pick the best model rather than the manual approach): 
```
Epoch 34/50
32384/32372 [==============================] - 250s - loss: 0.0105 - rmse: 0.1006 - val_loss: 0.0098 - val_rmse: 0.0962
Epoch 35/50
32384/32372 [==============================] - 251s - loss: 0.0102 - rmse: 0.0993 - val_loss: 0.0100 - val_rmse: 0.0970
Epoch 36/50
32384/32372 [==============================] - 250s - loss: 0.0102 - rmse: 0.0987 - val_loss: 0.0097 - val_rmse: 0.0952
Epoch 37/50
32384/32372 [==============================] - 250s - loss: 0.0102 - rmse: 0.0992 - val_loss: 0.0098 - val_rmse: 0.0964
Epoch 38/50
32384/32372 [==============================] - 249s - loss: 0.0101 - rmse: 0.0988 - val_loss: 0.0099 - val_rmse: 0.0967
Epoch 39/50
32384/32372 [==============================] - 250s - loss: 0.0102 - rmse: 0.0989 - val_loss: 0.0099 - val_rmse: 0.0966
Epoch 40/50
32384/32372 [==============================] - 249s - loss: 0.0101 - rmse: 0.0984 - val_loss: 0.0096 - val_rmse: 0.0951
Epoch 41/50
32384/32372 [==============================] - 249s - loss: 0.0099 - rmse: 0.0978 - val_loss: 0.0097 - val_rmse: 0.0958
Epoch 42/50
32384/32372 [==============================] - 249s - loss: 0.0100 - rmse: 0.0980 - val_loss: 0.0098 - val_rmse: 0.0961
Epoch 43/50
32384/32372 [==============================] - 249s - loss: 0.0099 - rmse: 0.0975 - val_loss: 0.0097 - val_rmse: 0.0959
Epoch 44/50
32384/32372 [==============================] - 249s - loss: 0.0099 - rmse: 0.0979 - val_loss: 0.0096 - val_rmse: 0.0952
Epoch 45/50
32384/32372 [==============================] - 249s - loss: 0.0099 - rmse: 0.0977 - val_loss: 0.0098 - val_rmse: 0.0962
Epoch 46/50
32384/32372 [==============================] - 250s - loss: 0.0099 - rmse: 0.0974 - val_loss: 0.0095 - val_rmse: 0.0944
Epoch 47/50
32384/32372 [==============================] - 249s - loss: 0.0099 - rmse: 0.0975 - val_loss: 0.0096 - val_rmse: 0.0949
Epoch 48/50
32384/32372 [==============================] - 249s - loss: 0.0097 - rmse: 0.0964 - val_loss: 0.0096 - val_rmse: 0.0952
Epoch 49/50
32384/32372 [==============================] - 250s - loss: 0.0098 - rmse: 0.0972 - val_loss: 0.0097 - val_rmse: 0.0955
Epoch 50/50
32384/32372 [==============================] - 249s - loss: 0.0098 - rmse: 0.0973 - val_loss: 0.0095 - val_rmse: 0.0945/home/nalapati/anaconda3/envs/sdc-behavioral-cloning/lib/python3.5/site-packages/keras/engine/training.py:1527: UserWarning: Epoch comprised more than `samples_per_epoch` samples, which might affect learning results. Set `samples_per_epoch` correctly to avoid this warning.
  warnings.warn('Epoch comprised more than '
INFO:__main__:Wrote final model to /home/nalapati/udacity/sdc/udacity-p3/models/1484977600.h5
INFO:__main__:Evaluation: [0.0096666271148535664, 0.098319006885004515]
INFO:__main__:Baseline MSE 0.02549, training MSE 0.00967, improvement 62.08%
INFO:__main__:output config: {'model_uri': '/home/nalapati/udacity/sdc/udacity-p3/models/1484977600.h5'}
```
## Results
* The model that solves track 1 is in the repo: **(model.json, model.h5)**. Dataset that solves track 1 can be downloaded at: (https://s3-us-west-1.amazonaws.com/sdc-datasets/dataset_track_1.zip) [Needs Permissions]. The final model shows really good track recovery. 
* I also solved Track 2 in isolation but haven't merged the two datasets yet to make a generalized model. The model that solves Track 2 is in the repo: **(track_2_model/model.json, track_2_model/model.h5)**. Dataset that solves track 2 can be downloaded at: (https://s3-us-west-1.amazonaws.com/sdc-datasets/dataset_track_2.zip) [Needs Permissions]. **NOTE: If you plan on testing track 2, the throttle in drive.py needs to be updated to 0.3 since the track has a lot of uphill sections.**

## Running the code.

```
# clone the repo
# If you don't already have deps run:
# conda env create -f environment.yml
# source activate sdc-behavioral-cloning
# kick off the simulator and switch to autonomous + track 1
python drive.py model.json
```
