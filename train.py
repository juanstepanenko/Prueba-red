import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import applications

def model(amountClasses):
    vgg=applications.vgg16.VGG16()
    cnn=Sequential()
    for layer in vgg.layers:
        cnn.add(layer)
    cnn.layers.pop()
    for layer in cnn.layers:
        layer.trainable=False
    cnn.add(Dense(amountClasses,activation='softmax'))
    
    return cnn

K.clear_session()

trainingData = './data/training'
validationData = './data/validation'

epochs=20 # Number of epochs ("itertions") of the the data set in training
length, height=224, 224 # Image size on pixels
batchSize=32 # Number of pictures to be processed in each step
steps=1000 # Number of batches proccesed in one iteration
validationSteps=300 # Number of steps at the end of each iteration    
conv1Filters=32 # Picture Depth
conv2Filters=64 # Picture Depth
filter1Size=(3, 3)
filter2Size=(2, 2)
poolSize=(2, 2) # Filter size used in the maxpooling
classes=2 # Number of problems
lr=0.0004 # Learning rate 


## Pictures preparation (pre-processing information)

datagenTraining=ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0, # This parameter rotates the diferent pictures (oj)
    zoom_range=0.2, 
    horizontal_flip=False) # Flips the picture

datagenValidation=ImageDataGenerator(rescale=1. / 255)

trainingImage=datagenTraining.flow_from_directory(
    trainingData,
    target_size=(length, height),
    batch_size=batchSize,
    class_mode='categorical') # We use this class mode when there are no classes/problem combinations

validationImage=datagenValidation.flow_from_directory(
    validationData,
    target_size=(length, height),
    batch_size=batchSize,
    class_mode='categorical')


# Network creation

cnn=model(classes)

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])


cnn.fit_generator(
    trainingImage,
    steps_per_epoch=steps,
    epochs=epochs,
    validation_data=validationImage,
    validation_steps=validationSteps)

target_dir = './model/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./model/model.h5')
cnn.save_weights('./model/weigths.h5')