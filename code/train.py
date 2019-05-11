import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, array_to_img
from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Flatten
from keras.metrics import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.initializers import lecun_uniform
from keras import initializers, regularizers
from keras.utils import print_summary, plot_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
# to array and store in data list

def get_x_y(label_encoder,onehot_encoder,paths,tot):
    x = []
    y = []
    for img_pth in paths:
        h, word = os.path.split(os.path.dirname(img_pth))
        image = cv2.imread(img_pth,0)
        ret,image=cv2.threshold(image,200,255,cv2.THRESH_BINARY)
        im = img_to_array(image)
        x.append(im)
        y.append(word)

    x = np.array(x, dtype="float")/255
    y = np.array(y)
    if tot==1:
        y = label_encoder.fit_transform(y)
        y = y.reshape(len(y), 1)
        y = onehot_encoder.fit_transform(y)
        np.save('classes.npy',label_encoder.classes_)
    else:
        y = label_encoder.transform(y)
        y = y.reshape(len(y), 1)
        y = onehot_encoder.transform(y)       
    return label_encoder,onehot_encoder,x, y

train_image_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk("database/train") for f in filenames if os.path.splitext(f)[1] == '.png']
test_image_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk("database/test") for f in filenames if os.path.splitext(f)[1] == '.png']

label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False,categories='auto')

label_encoder,onehot_encoder,x_train, y_train = get_x_y(label_encoder,onehot_encoder,train_image_paths,1)
label_encoder,onehot_encoder,x_test, y_test = get_x_y(label_encoder,onehot_encoder,test_image_paths,0)

model = Sequential()
model.add(Conv2D(128,(5,5), kernel_initializer='lecun_uniform', activation='relu', data_format="channels_last", input_shape=(128,128,1)))
model.add(Conv2D(128,(5,5), kernel_initializer='lecun_uniform', activation='relu',kernel_regularizer=regularizers.L1L2(l1=0.0, l2=0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128,(3,3), kernel_initializer='lecun_uniform', activation='relu',kernel_regularizer=regularizers.L1L2(l1=0.0, l2=0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256,(3,3), kernel_initializer='lecun_uniform', activation='relu',kernel_regularizer=regularizers.L1L2(l1=0.0, l2=0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256,(3,3), kernel_initializer='lecun_uniform', activation='relu',kernel_regularizer=regularizers.L1L2(l1=0.0, l2=0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation=None, kernel_initializer='lecun_uniform'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation=None, kernel_initializer='lecun_uniform'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation=None, kernel_initializer='lecun_uniform'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

model.compile(Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
earlystopper = EarlyStopping(monitor='val_acc', patience=3, verbose=1)


with open("cnn_architecture.txt","w") as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))



history = model.fit(x_train,y_train, epochs=50, callbacks=[tensorboard,earlystopper], batch_size=8, validation_data=(x_test,y_test))
model.save('handwritten_word_recognition.h5'.format())

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_plot.jpg')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_plot.jpg')


results = model.evaluate(x_train,y_train)
print(results)

results = model.evaluate(x_test,y_test)
print(results)

predictions= model.predict(x_test)
for i in range(len(predictions)):
    print(label_encoder.inverse_transform([np.argmax(y_test[i])]),label_encoder.inverse_transform([np.argmax(predictions[i])]))
