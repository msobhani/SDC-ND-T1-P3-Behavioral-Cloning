import sys
import csv
import cv2
import argparse
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

def preprocess_image(image):
    
    
    # convert the colors
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # apply subtle blur
    new_image = cv2.GaussianBlur(new_image, (3,3), 0)
    
    # crop
    new_image = new_image[60:130, :]
    
    # resize
    new_image = cv2.resize(new_image,(160, 70), interpolation = cv2.INTER_AREA)


    
    return new_image

def process_batch(input_sample, correction_factor):
    
    steering_angle = np.float32(input_sample[3])
    images, steering_angles = [], []

    for image_path_index in range(3):
        image_location = input_sample[image_path_index].split('/')[-1]      
        image = cv2.imread(image_location)     
        
        image_new = preprocess_image(image)
        images.append(image_new)

        if image_path_index == 0:
            steering_angles.append(steering_angle)
        elif image_path_index == 1:
            steering_angles.append(steering_angle + correction_factor)
        elif image_path_index == 2:
            steering_angles.append(steering_angle - correction_factor)            
        
        # Flip the center image and add to the dataset
        if image_path_index == 0:
            flipped_center_image = cv2.flip(image_new, 1)
            images.append(flipped_center_image)
            steering_angles.append(-steering_angle)

    return images, steering_angles
    
def data_generator(input_data, correction_factor, batch_size):

    input_len = len(input_data)

    while True:
        
        shuffle(input_data)

        for offset in range(0, input_len, batch_size):

            images, steering_angles = [], []
            batch_samples = input_data[offset:offset + batch_size]


            for sample in batch_samples:        
                augmented_images, augmented_angles = process_batch(sample, correction_factor)
                images.extend(augmented_images)
                steering_angles.extend(augmented_angles)

            X_train, y_train = np.array(images), np.array(steering_angles)
            yield shuffle(X_train, y_train)
         
        
##### Main code #####

EPOCHS = 2
STEERING_CORRECTION_FACTOR = 0.2
BATCH_SIZE = 128


parser = argparse.ArgumentParser(description='Trains a self-driving car')
parser.add_argument(
    '--input-data-path',
    type=str,
    default='./data',
    help='Provide path to the training dataset, which includes the images and the driving log'
)

args = parser.parse_args()

driving_log_path = args.input_data_path + '/driving_log.csv'

driving_data = []
# Import data
with open(driving_log_path) as csvlogfile:
    reader = csv.reader(csvlogfile)

    # Avoid the first row
    next(reader)

    for line in reader:
        driving_data.append(line)

#Split the data
train, validation = train_test_split(driving_data, test_size=0.2)

#Create the train and validation generators
train_generator = data_generator(train, STEERING_CORRECTION_FACTOR, BATCH_SIZE)
validation_generator = data_generator(validation, STEERING_CORRECTION_FACTOR, BATCH_SIZE)

#Create the model
model = create_model()

model.fit_generator(generator=train_generator,
                     validation_data=validation_generator,
                     epochs=EPOCHS,
                     steps_per_epoch=len(train) * 2,
                     validation_steps=len(validation))

#Save the model
model.save('model.h5')