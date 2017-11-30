import numpy as np
import pandas as pd
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D, Flatten, Input, Dropout, Conv2D, LSTM, Merge, Reshape
from keras.models import load_model
from keras.optimizers import Adam

np.random.seed(123)

# image dimentions
img_x = 32
img_y = 14

# total features: 32x14 = 448

# reading training data
df = pd.read_csv('data.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
df = df[df[str(448)]!='S']
df = df[df[str(448)]!='s']

# dividing features and labels
X = df.drop(str(448), axis=1)
y = df[str(448)]

# resizing training data
X_data = X.values.reshape(-1,448)
X_data = X_data.reshape(X_data.shape[0], 14, 32, 1)
X_data = X_data.astype('float32')
X_data /= 255

# categorizing labels
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
y_data = to_categorical(y, 3)

# split train, test and validation data
X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, random_state=0, test_size=0.1)
X_test, X_val, y_test, y_val = train_test_split(X_test,y_test, random_state=0, test_size=0.5)

# creating model
model = Sequential()
 
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(14,32,1)))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# training model
model.fit(X_train, y_train, batch_size=150, epochs=15, verbose=1, validation_data=(X_val,y_val))

# getting accuracy
scores = model.evaluate(X_test, y_test, verbose=0)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))