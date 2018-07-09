# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview
---

The project is submitted with the following five files: 
* `model.py` (script used to create and train the model)
* `drive.py` (script to drive the car - feel free to modify this file)
* `model.h5` (a trained Keras model)
* `README.md` a report writeup file (markdown)
* `video.mp4` (a video recording of the vehicle driving autonomously around the track for at least one full lap)

The goals of the project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Model Architecture and Training Strategy

### 1.Model architecture

My model is based on the recommended Nvidia architecture for the self-driving cars. The first layer of the model normalizes the data using a lambda layer. Then, several `Conv2d` layers with different filter sizes from 3x3 to 5x5 compose the next level of layers. To add nonlinearity `RELU` layers are added in the end. The Nvidia architecture is depicted as follows:

![Net](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

The Keras model is created in `create_model()` function:

```python
def create_model(loss='mse', optimizer='adam'):
    
    model = Sequential()
    
    # Normalization Layer
    model.add(Lambda(lambda x:  (x / 127.5) - 1., input_shape=(70, 160, 3)))
    
    #Convolution layers with activations
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))

    # Flatten Layers
    model.add(Flatten())
    
    # Fully Connected Layers
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    
    #Output Layer
    model.add(Dense(1))

    # Apply the given optimizer and loss function
    model.compile(loss=loss, optimizer=optimizer)

    return model
```


### 2.Preprocessing

The image input from the `drive.py` is in the RGB color space. On the other hand, the OpenCV reads the images in the BGR color space. Therefore I've decided to train the model on RGB images, which required the conversion of the OpenCV output images to RGB. 

In the next step, a gaussian blur was applied to the input image to reduce the noise.Then, the resulting image was cropped to remove the hood and the sky, and was finally resized to match the input plane.

To create an augmented dataset, I've used the center, left and right images as well as a flipped center image. 

### 3.Reducing the overfitting

The training and validation sets were obtained by splitting the input dataset by the factor of 0.2 (80% Training vs 20% Validation). To reduce the overfitting, the dataset was augmented by using the center, left and right images as well as flipping the center image. This helped to generlize the model and reduce the chance of overfitting.

### 4.Model parameters 

Because an Adam optimizer was used in the Keras model, there was no need to manually define the learning rate.

### 5.Training data

The training data was collected by running the simulator, and driving the car in the ego lane. This has included the center, left and right images. Data augmentation was used to enlarge the dataset.

The model was trained on my laptop for 2 Epochs, that took 2 hours to finish. Finally, the trained model was saved to be used by the simulator. Here is the output:

```sh
$>python model.py --input-data-path=./recorded_data/1
Using TensorFlow backend.
Epoch 1/2
2018-07-08 21:53:34.248402: I T:\src\github\tensorflow\tensorflow\core\platform\
cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow bi
nary was not compiled to use: AVX2
1482/1482 [==============================] - 3356s 2s/step - loss: 0.0032 - val_loss: 0.0192
Epoch 2/2
1482/1482 [==============================] - 3305s 2s/step - loss: 2.1125e-04 -val_loss: 0.0194
```

### 6.Result
Using the simulator in the autonomous mode, and using the trained model, the car was able to drive and stay on the road during the first and second track. This is recorded in the `video.mp4` for consideration.