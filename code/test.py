import os
import sys
import cv2
from keras.models import Sequential, load_model
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from PIL import Image
import numpy as np
import tensorflow as tf

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy')

m,n = 128,128

saved_model = "handwritten_word_recognition.h5"
model = load_model(os.path.join(os.getcwd(),saved_model))
test_image = sys.argv[1]
print(test_image)
desired_size = 666
im = Image.open(test_image)
im.show()
x=[]
image = cv2.imread(test_image,0)
resized = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)
ret,image=cv2.threshold(resized,127,255,cv2.THRESH_BINARY)
im = img_to_array(image)
x.append(im)
x = np.array(x, dtype="float")/255.0

predictions = model.predict(x)
for i in range(len(predictions)):
    print(label_encoder.inverse_transform([np.argmax(predictions[i])]))
