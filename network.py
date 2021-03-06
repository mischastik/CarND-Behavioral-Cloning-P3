import csv
import cv2
import numpy as np
import sklearn
from random import shuffle

samples = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    # we divide the batch_size by six to account for augmentations which effectively provide six samples for one entry
    batch_size = batch_size // 6
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                # load center, left and right image
                for i in range(3):
                    source_path = batch_sample[i]
                    image = cv2.imread(source_path)
                    images.append(image)
                # load the steering wheel angle
                measurement = float(batch_sample[3])
                measurements.append(measurement)
                # apply static correction angles for left and right image
                measurements.append(measurement + 0.2)
                measurements.append(measurement - 0.2)
            # augment the data-set with flipped images
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)


            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            #X_train = np.array(images)
            #y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.backend import tf as ktf
from keras.models import load_model
from pathlib import Path

my_file = Path("model.h5")
# if we already have a trained model, load it and continue training
# important: Remember to delete model.h5 if the model architecture is changes, otherwise changes won't take effect
if my_file.exists():
    model = load_model('model.h5', custom_objects={"ktf": ktf})
    model.fit_generator(train_generator, steps_per_epoch=len(train_samples), epochs=2, validation_data=validation_generator,
                    validation_steps=len(validation_samples))
    model.save('model.h5')

else:
    # we have no saved model, so build the model and train it, then save it.
    # the model is inspired by the model suggested by NVidia with some additional convolutional layers.
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(((65, 25), (0, 0))))
    model.add(Lambda(lambda x: ktf.image.resize_images(x, (68, 200))))
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    model.add(Convolution2D(24, (5, 5), activation="relu"))
    model.add(Convolution2D(24, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(36, (5, 5), activation="relu"))
    model.add(Convolution2D(36, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(48, (5, 5), activation="relu"))
    model.add(Convolution2D(48, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(Convolution2D(64, (3, 3), activation="relu", padding="valid"))
    model.add(Flatten())
    model.add(Dense(100))
    # dropout didn't help in this case, so we deactivated it
    #model.add(Dropout(0.5, noise_shape=None, seed=None))
    model.add(Dense(50))
    #model.add(Dropout(0.5, noise_shape=None, seed=None))
    model.add(Dense(10))
    #model.add(Dropout(0.5, noise_shape=None, seed=None))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, steps_per_epoch=len(train_samples), epochs=1, validation_data=validation_generator,
                    validation_steps=len(validation_samples))
    model.save('model.h5')
