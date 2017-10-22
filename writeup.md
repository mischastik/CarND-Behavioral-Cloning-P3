**Behavioral Cloning** 

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model_architecture.png "Model Visualization"

Rubric Points
---
**Files Submitted & Code Quality**

1. Submission includes all required files and can be used to run the simulator in autonomous mode

The project includes the following files:
* network.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (unchanged)
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* run1.mp4 is a video showing a successful autonomous drive of one lap of track one

2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

3. Submission code is usable and readable

The network.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

**Model Architecture and Training Strategy**

1. An appropriate model architecture has been employed

The model is based on the model suggested by NVidia with some additional convolution layers. I re-interpolated the images to match the dimensions expected by the NVidia model. Model also contains cropping the top and bottom to remove regions that don't contain relevant information.
Throughout the network, ReLU activations are used and the convolution kernels are mostly 5x5.

2. Attempts to reduce overfitting in the model

I used a training / validation split of 0.8 to 0.2 with no test data set as testing is done on the actual simulator.
To avoid overfitting a provided training data with a wide variety driving situations.
I also "supervised" the training by looking at the results at the end of the epoch an then deciding on continuing or stopping the training. The model is saved after each epoch and a restart of the script performs another epoch of training.

I also added dropout layers between the dense layers, but I couldn't observe any positive effects, so these layers are deactivated.

The training and validation errors are very similar, so the trained model show no obvious signs of overfitting.

3. Model parameter tuning

The model uses an adam optimizer, so the learning rate was not tuned manually (network.py line 104). The number of epochs was tuned by manually stoping the training when no further improvements could be observed.

4. Appropriate training data

I started with a training set of one lap of center driving and iteratively increased the training data size and content until a good overall result was achieved.

The final data set consists of driving situations such as
* two laps of center driving,
* driving the track in the opposite direction,
* many different recovery situations for off-center situations,
* additional data for non-standard scenarios such a the bridge or very sharp turns.

**Development Strategy**

1. Solution Design Approach

The first attempt was a simple model with normalization, cropping, a few convolutional layers followed by a dense layer trained on a simple dataset of just one lap of standard driving. I went through several iterations of reviewing the performance, generating more training data and adding complexity which finally led to the final model.
I used the model suggested by NVidia as a basis and I re-interpolated the images to match the dimensions expected by the NVidia model. The performance was not sufficient though, so I added some additional convolutional layers with same-size-padding to keep the original dimensions.
The images are also cropped at the top and bottom to remove regions that don't contain relevant information.
Throughout the network, ReLU activations are used and the convolution kernels are mostly 5x5.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

2. Final Model Architecture

The final model architecture is depicted here:

![Model architecture][image1]

3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving and an additional lap of center lane driving in the opposite direction.

Next I recorded some samples of recovering when getting of center and additional samples for particualarly crucial scenarios like the bridge and very sharp turns and sections where the lane markings were missing.  

To augment the data set, I also flipped images and angles and also used the left and right camera images with a fixed angle correction value.

After the collection and augmentation process, I had approx. 20000 data points. A generator is used to feed the tarining data to the optimizer.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I inspected the training and validation results after each epochs and stopped after three epochs since there was only little additional progress in training and validation error metrics.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
