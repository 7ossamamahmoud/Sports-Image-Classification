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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, MaxPool2D, \
    Activation
from tensorflow.keras.optimizers import Adam
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import pandas as pd


IMG_SIZE = 224
BATCH_SIZE = 16
# Enter the path of your image data folder
DATA_PATH = '/content/drive/MyDrive/Data'


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
train_generator.class_indices


def VGG16():
    model = Sequential()
    model.add(Conv2D(input_shape=(IMG_SIZE, IMG_SIZE, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='vgg16'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu', name='fc1'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu', name='fc2'))
    model.add(Dropout(0.22))
    model.add(Dense(6, activation='softmax', name='output'))
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"])
    return model


model = VGG16()
Vgg16 = Model(inputs=model.input, outputs=model.get_layer('vgg16').output)

# Load weights
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

weights_path = tf.keras.utils.get_file(
    'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    WEIGHTS_PATH_NO_TOP,
    cache_subdir='models',
    file_hash='6d6bbae143d832006294945121d1f1fc')

Vgg16.load_weights(weights_path)
### Freezing the Weights ###
for layer in Vgg16.layers:
    layer.trainable = False

model.summary()

# autosave best Model
checkpoint_path = 'Data/vgg16_best_model.ckpt'

checkpoint_dir = os.path.dirname(checkpoint_path)
early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit_generator(train_generator,
                              epochs=100,
                              verbose=1,
                              validation_data=valid_generator,
                              callbacks=[checkpoint, early])

evl = model.evaluate_generator(valid_generator)
print("Loss: {:0.4f}".format(evl[0]), "Score: {:0.4f}".format(evl[1]))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(len(acc))


plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
plt.show()


model.save('/content/drive/MyDrive/NN23_VGG16-UPDACC84S-model.h5')


TEST_DIR ='/content/drive/MyDrive/Test'
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
test_df.to_csv(r'./VGG16_NN23.csv', index=False)

