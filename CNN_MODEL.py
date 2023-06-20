import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import random
import keras
from keras.models import Sequential, load_model
from random import shuffle
from keras_preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import random
import pickle
import cv2
import os
import imutils
from imutils import paths
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

IMG_SIZE = 224
IMG_HEIGHT = 224
IMG_WIDTH=244
BATCH_SIZE = 16
DATA_PATH = '/content/drive/MyDrive/Data'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1 / 255.0,
                                   rotation_range=30,
                                   zoom_range=0.4,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   shear_range=0.15,
                                   horizontal_flip=True,
                                   validation_split=0.1)

train_generator = train_datagen.flow_from_directory(DATA_PATH,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    target_size=(IMG_SIZE, IMG_SIZE),
                                                    subset='training')

valid_generator = train_datagen.flow_from_directory(DATA_PATH,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    target_size=(IMG_SIZE, IMG_SIZE),
                                                    subset='validation')
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,MaxPool2D
warnings.filterwarnings('ignore')

def cnn():
      model = Sequential()
      model.add(
          Conv2D(input_shape=(IMG_SIZE, IMG_SIZE, 3), filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
      model.add(MaxPool2D(pool_size=(2, 2)))
      model.add(Conv2D(input_shape=(IMG_SIZE, IMG_SIZE, 3), filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
      model.add(MaxPool2D(pool_size=(2, 2)))
      model.add(Conv2D(input_shape=(IMG_SIZE, IMG_SIZE, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
      model.add(Conv2D(input_shape=(IMG_SIZE, IMG_SIZE, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
      model.add(MaxPool2D(pool_size=(2, 2)))
      model.add(Conv2D(input_shape=(IMG_SIZE, IMG_SIZE, 3), filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
      model.add(Conv2D(input_shape=(IMG_SIZE, IMG_SIZE, 3), filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
      model.add(MaxPool2D(pool_size=(2, 2)))
      model.add(Conv2D(input_shape=(IMG_SIZE, IMG_SIZE, 3), filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
      model.add(Conv2D(input_shape=(IMG_SIZE, IMG_SIZE, 3), filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
      model.add(Conv2D(input_shape=(IMG_SIZE, IMG_SIZE, 3), filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
      model.add(MaxPool2D(pool_size=(2, 2),name='vgg16'))
      model.add(Flatten())
      model.add(Dense(512, activation='relu', name='fc1'))
      model.add(Dense(512, activation='relu', name='fc2'))
      model.add(Dense(6, activation='softmax', name='output'))    
      return model
          
model = cnn()
model.summary()
from tensorflow.keras.optimizers import Adam

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# autosave best Model
checkpoint_path = '/content/drive/MyDrive/Test/cnn_best_model.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
from tensorflow.keras.models import load_model
history = model.fit_generator(train_generator,
                              epochs=100,
                              verbose=1,
                              validation_data=valid_generator,
                              callbacks=[checkpoint, early])

  
evl = model.evaluate_generator(valid_generator)
print("Loss: {:0.4f}".format(evl[0]), "Accuracy: {:0.4f}".format(evl[1]))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(len(acc))
model.save('/content/drive/MyDrive/CNN-Model.h5')

import pandas as pd
TEST_DIR = '/content/drive/MyDrive/Test'
img_path = os.listdir(TEST_DIR)
test_df = pd.DataFrame({'image_name': img_path})
n_test_samples = test_df.shape[0]
print("Number of Loaded Test Data Samples: ", n_test_samples)
test_datagen = ImageDataGenerator(rescale=1 / 255.0)

test_generator = test_datagen.flow_from_dataframe(test_df,
                                                  directory=TEST_DIR,
                                                  x_col='image_name',
                                                  target_size=(IMG_SIZE, IMG_SIZE),
                                                  y_col=None,
                                                  batch_size=1,
                                                  class_mode=None,
                                                  shuffle=False)
import numpy as np
pred_array = model.predict(test_generator, steps=np.ceil(n_test_samples / 1.0))
predictions = np.argmax(pred_array, axis=1)
test_df['label'] = predictions
test_df.head()
test_df.to_csv(r'./CNN_NN23.csv', index=False)