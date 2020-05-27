import pandas as pd # data analysis toolkit - create, read, update, delete datasets
import numpy as np #matrix math
from sklearn.model_selection import train_test_split #to split out training and testing data 
#keras is a high level wrapper on top of tensorflow (machine learning library)
#The Sequential container is a linear stack of layers
from keras.models import Sequential
#popular optimization strategy that uses gradient descent 
from keras.optimizers import Adam
#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
#what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
#helper class to define input shape and generate training images given image paths & steering angles

#for command line argumen
#for reading files
import os

import cv2, os
import numpy as np
import matplotlib.image as mpimg

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread("IMG/"+image_file)


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.5
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:

        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    
    x1, y1 = image.shape[1] * np.random.rand(), 0
    x2, y2 = image.shape[1]* np.random.rand(), image.shape[0]
    xm, ym = np.mgrid[0:image.shape[0], 0:image.shape[1]]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(data_dir, center, left, right, steering_angle, range_x = 100, range_y = 10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle
def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = []
    steers = []
    i = 0
    for index in np.random.permutation(image_paths.shape[0]):
        center, left, right = image_paths[index]
        steering_angle = steering_angles[index]
            # argumentationss
        if is_training and np.random.rand() < 0.8:
            image, steering_angle = augument(data_dir, center, left, right, steering_angle)
        else:
            image = load_image(data_dir, center)
            # add the image and steering angle to the batch
        images.append(preprocess(image))
        steers.append(steering_angle)
        i += 1
        if i == image_paths.shape[0]:
            break
    return images , steers
         
def load_data():
    """
    Load training data and split it into training and validation set
    """
    
    data_df = pd.read_csv(('driving_log.csv'), names = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

   
    X = data_df[['center', 'left', 'right']].values
    
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_valid, y_train, y_valid
def build_model():
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, kernel_size = (5, 5), activation='elu', strides = (2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(36, kernel_size = (5, 5), activation='elu', strides = (2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(48, kernel_size = (5, 5), activation='elu', strides = (2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size = (3, 3) , activation='elu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size = (3, 3), activation='elu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, activation ='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation ='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation ='elu'))
    model.add(Dense(1))
    model.summary()

    return model

def train_model(model, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    
    checkpoint = ModelCheckpoint('model.h5',
                                 monitor = 'val_loss',
                                 verbose = 0,
                                 save_best_only = True,
                                 mode = 'auto')

    #calculate the difference between expected steering angle and actual steering angle
    #square the difference
    #add up all those differences for as many data points as we have
    #divide by the number of them
    #that value is our mean squared error! this is what we want to minimize via
    #gradient descent
    model.compile(loss = 'mean_squared_error', optimizer = Adam(lr = 1.0e-4))

    #Fits the model on data generated batch-by-batch by a Python generator.

    #The generator is run in parallel to the model, for efficiency. 
    #For instance, this allows you to do real-time data augmentation on images on CPU in 
    #parallel to training your model on GPU.
    #so we reshape our data into their appropriate batches and train our model simulatenously
    batch_sizes = 64
    X_train,y_train = batch_generator("IMG/", X_train, y_train,batch_sizes, True)
    epoch = 50
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_valid,y_valid = batch_generator("IMG/", X_valid, y_valid,batch_sizes, False)
    X_valid = np.asarray(X_valid)
    y_valid = np.asarray(y_valid)
    model.fit(X_train,y_train, epochs = epoch,max_queue_size=1,validation_data = (X_valid,y_valid),batch_size = batch_sizes,callbacks=[checkpoint],verbose = 1)


                       
X_train, X_valid, y_train, y_valid = load_data()

for i in range(len(X_train)):
   a,b,c = X_train[i]
   a1 = a.split('/')
   b1 = b.split('/')
   c1 = c.split('/')
   X_train[i][0] = a1[6]
   X_train[i][1] = b1[6]
   X_train[i][2] = c1[6]

for i in range(len(X_valid)):
   a,b,c = X_valid[i]
   a1 = a.split('/')
   b1 = b.split('/')
   c1 = c.split('/')
   X_valid[i][0] = a1[6]
   X_valid[i][1] = b1[6]
   X_valid[i][2] = c1[6]
model = build_model()
model.save("model.h5")
train_model(model,X_train, X_valid, y_train, y_valid )
model.save("model.h5")
