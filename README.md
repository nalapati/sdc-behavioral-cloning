Using Deep Learning to Clone Driving Behavior
---------------------------------------------

NOTE: This project builds on work done in https://medium.com/udacity/teaching-a-machine-to-steer-a-car-d73217f2492c, specifically Ilya Edrenkin's approach of using a 3D Convolutional Neural Network.

Step 0: Infrastructure Setup
As part of challenge 2 we got to a really good infrastructure setup on https://github.com/emef/sdc. This submission is a cut down version of the sdc project.

Step 1: Model design
Requirements:
a. The problem being solved here is given the center camera image from a car, predict the steering angle (fixed throttle). 
b. This problem does not require scale or rotation or translation invariance.
c. This problem requires generalization to different roads and lighting conditions.
d. Requires recovering from incorrect predictions to stay on the road and not go off track.

In view of the above, this solution implements a model that predicts the steering angle in radians given an input image
from the center camera of a car. It is sufficient to go shallow and not use a lot of non-linearity by using just a few
convolutional layers, but going wide by collecting a LOT of data and using a large number of filters in a few layers.
The 3D convolutional layers in the model allow learning time series information across different frames basically
the depth dimension amounts to timesteps or camera images from previous points in time.

The model architecture is as follows:
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-behavioral-cloning/master/images/model.jpg)

Step 2: Training on the sample data provided
The first step once the infrastructure was setup was to train the model on the sample data to determine if I could make
some predictions (any predictions), and also set it up so I had the basic metrics flowing through and models working 
end to end and used to drive the car.

Step 3: Collecting data
Once the pipeline was setup end to end the approach was to start collecting data. I drove the car two and fro twice on
track 1 for the first dataset.

Step 4: Overfitting the collected data to solve Track 1
I tuned the model (number of layers, convolutional filters, filter sizes, hidden layer sizes) to come to a solution
that solved the track 1 test set, but the model overfit(later fixed to add dropouts + l2 regularization) in that it
would sway out after 2 laps, the data didn't have any correction samples. 

Step 5: Collecting additional data
This was largely manual, but surprisingly required a lot of discipline to manage these datasets, I added increments to
a core dataset (each with its READMEs to track additions), My final dataset has the following composition:
a. Track 1 to and fro 9 times.
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-behavioral-cloning/master/images/center_2017_01_18_12_56_26_395.jpg)
b. Track 1 correction to and fro 9 times.
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-behavioral-cloning/master/images/center_2017_01_19_07_23_58_402.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-behavioral-cloning/master/images/center_2017_01_19_07_23_59_149.jpg)
c. Track 1 sharp turns to and fro 12 times.
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-behavioral-cloning/master/images/center_2017_01_19_09_40_52_539.jpg)
d. Track 1 data on the bridge section.
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-behavioral-cloning/master/images/center_2017_01_19_09_40_46_616.jpg)

The model that solves track 1 is in the repo:
- 
Dataset that solves track 1 can be downloaded at:
- 

The final model shows really good track recovery. 

I also solved Track 2 in isolation but haven't merged the two datasets yet. The model that solves Track 2 is at (0.3 throttle since there are uphill sections on Track 2):
- 
Dataset that solves track 2 can be downloaded at:
- 
