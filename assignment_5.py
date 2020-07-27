# -*- coding: utf-8 -*-
"""assignment 5.ipynb

"""


import keras
import tensorflow as tf
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

#3)
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
plt.show()
training_images[0]
training_labels[0]

print(training_images.shape)
test_images.shape

#4) Nomalize the training and testing data
training_images = training_images.reshape((60000, 28 * 28))
training_images = training_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#5)Create a DNN 
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#encode the training and testing labels
from keras.utils import to_categorical

training_labels = to_categorical(training_labels)
test_labels = to_categorical(test_labels)
print(training_labels.shape)
print(training_labels[0])

#training accuracy is 92.52%
network.fit(training_images, training_labels, epochs=12, batch_size=128)

network.summary()

#test_acc is 88.84%
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

"""Question2: 

Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad. 

    !wget --no-check-certificate \
         "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
          -O "/tmp/happy-or-sad.zip"

1. Load and nomalize the input data
2. Create a convolutional neural network that trains to 99.9% accuracy on these images. 
3. Upload and print 10 of your own faces, half happy and half sad.(Yes, your own faces!)
4. Test the 10 pics using the model you trained and print out the acurry rate from the model.

"""

#Question 2 
#1)


import os
import zipfile

local_zip = '/tmp/happy-or-sad.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp/happy-or-sad')
zip_ref.close()

# Directory with our training chappy/sad pictures
train_happy_dir = os.path.join('/tmp/happy-or-sad','happy')
train_sad_dir = os.path.join('/tmp/happy-or-sad','sad')

train_happy_fnames = os.listdir(train_happy_dir)
train_sad_fnames = os.listdir(train_sad_dir)
print(train_happy_fnames[:10])
print(train_sad_fnames[:10])

#nomalize data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
# Flow training images in batches of 20 using generator

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('/tmp/happy-or-sad',
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(300, 300))

# Commented out IPython magic to ensure Python compatibility.
###see the picture
# %matplotlib inline

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over images
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_happy_pix = [os.path.join(train_happy_dir, fname) 
                for fname in train_happy_fnames[ pic_index-8:pic_index] 
               ]

next_sad_pix = [os.path.join(train_sad_dir, fname) 
                for fname in train_sad_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_happy_pix+next_sad_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

#2)
#define the model
import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300*300 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results into a one dimension data to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    # Note that because we are facing a two-class classification problem, i.e. a binary classification problem, we will 
    # end our network with a sigmoid activation, so that the output of our network will be a single scalar between 0 and 1,
    # encoding the probability that the current image is class 1 (as opposed to class 0).
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.0001), #learning rate of 0.0001. 
              loss='binary_crossentropy', #binary_crossentropy loss, because it's a binary classification problem and our final activation is a sigmoid.
              metrics = ['acc'])

#training 
history = model.fit_generator(train_generator,
                              epochs=10,
                              verbose=1)
##accuracy is 1

#3)4)
import numpy as np

from google.colab import files
from keras.preprocessing import image

uploaded=files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path='/content/' + fn
  img=image.load_img(path, target_size=(300, 300))
  
  x=image.img_to_array(img)
  x=np.expand_dims(x, axis=0)
  images = np.vstack([x])
  
  classes = model.predict(images, batch_size=10)
  
  print(classes[0])
  
  if classes[0]>0:
    print(fn + " I am sad")
    
  else:
    print(fn + " I am happy!")

## singel number represent happy; double represent sad
## the accuracy is 0.6



ACCURACY_THRESHOLD = 0.995
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('acc') > ACCURACY_THRESHOLD):   
          print("Reached desired accuracy so cancelling training!" )   
          self.model.stop_training = True

callbacks = myCallback()
model.fit(train_generator, epochs=10, callbacks=[callbacks])