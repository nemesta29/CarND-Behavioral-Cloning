# **Behavioral Cloning** 

## Project Writeup

### The following writeup file gives an insight into the thought process and experimentation that went into developing the project for Behavioral Cloning

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 6 and 64 (model.py lines 52-69) 

The model includes RELU layers to introduce nonlinearity (model.py line 59, line 66, line 68), and the data is normalized in the model using a Keras lambda layer (code line 53). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 61 and 65). The specific positions were chosen by running the model on track and comparing to accuracy metrics and best performance was found at the given lines.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 72 -> using model.fit()). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 71).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, which were obtained through Udacity sample data supplied with project resources. Images were cropped appropriately to include only the road details. The data was normalised using (x/255 - 0.5) to bring the data to a standard mean.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to accuractely predict steering angle depending on the road seen.

My first step was to use a convolution neural network model similar to the LeNet model. I thought this model might be appropriate because it was built to handle image data for learning. However, since the prediction was different in our requirement, the model had to be modified to suit the regression needs of present situation.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set a higher mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include dropouts to reduce the number of parameters. Further, I used left and right camera images to increase the data and allow the model to learn better. Max pooling was initially used after every convolution vto reduce the number of inputs, but it was seen that data was lost due to excessive usage of pooling and hence it was kept only after two convolution layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, specifically along the second last turn. To improve the model, I ran appended data for that specific portion by driving in simulator. Consesquently, I trained with increased epochs on the original data set. Improvements were very close, so I chose the higher epoch model to create my final model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 52-69) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					          		          |
| ----------------- |:---------------------------------------------------------:|
| Lambda            | Applies normalization through (x/255 - 0.5) 		          |
| Cropping          | Crops image to include road only ((70,20), (0,0))		      |
| Convolution    		| 2D Convolution of depth 6 with 3x3 filter    		          | 
| Max-Pooling    		| Max-Pooling done with a 2x2 filter size     		          |
| Convolution    		| 2D Convolution of depth 16 with 3x3 filter    		        | 
| Max-Pooling    		| Max-Pooling done with a 2x2 filter size     		          |
| Activation        | Activation layer using reLU                               |
| Convolution    		| 2D Convolution of depth 64with 3x3 filter    		          | 
| Dropout           | A dropout layer to reduce overfitting                     |
| Flatten           | Layer to flatten convolution output                       |
| Dense             | A fully connected layer with output of 64                 |
| Dense             | A fully connected layer with output of 128                |
| Activation        | Activation layer using reLU                               |
| Dense             | A fully connected layer with output of 64                 |
| Activation        | Activation layer using reLU                               |
| Dense             | A fully connected layer with output of 1 (Output Layer)   |



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
