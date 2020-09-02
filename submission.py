import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATADIR = r'C:\Users\thebh\KAGGLE\NMLOcontest\train\train'
CATEGORIES = ['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana', 'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine', 'Corn', 'Cucumber Ripe', 'Grape Blue', 'Kiwi', 'Lemon', 'Limes', 'Mango', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple', 'Plum', 'Pomegranate', 'Potato Red', 'Raspberry', 'Strawberry', 'Tomato', 'Watermelon']

print("initial printing")
training_data_list = []

def training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_name = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            training_data_list.append([img_array, class_name])


training_data()


x = []
y = []
for features, label in training_data_list:
    x.append(features)
    y.append(label)
x = np.array(x)
y = np.array(y)

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)

print(len(X_train))
print(X_train.shape)






# Normalize data
X_train = X_train / 100 #[(i)/250 for i in X_train]
X_test = X_test / 100 #[(i)/250 for i in X_test]
print(len(X_train[3]))



#model architecture
model = tf.keras.Sequential(layers=[
    tf.keras.layers.Conv2D(
        input_shape=(x.shape[1], x.shape[2], 1),  #
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        padding='same',
        activation='relu'
    ),

    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(2, 2),
        strides=1,
        padding='same',
        activation='relu'
    ),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=2
    ),

    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(2, 2),
        strides=1,
        padding='same',
        activation='relu'
    ),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(2, 2),
        strides=1,
        padding='same',
        activation='relu'
    ),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=2
    ),

    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(2, 2),
        strides=1,
        padding='same',
        activation='relu'
    ),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.BatchNormalization(),

    # tf.keras.layers.Conv2D(
    #     filters=16,
    #     kernel_size=(2, 2),
    #     strides=1,
    #     padding='same',
    #     activation='relu'
    # ),
    #   tf.keras.layers.Dropout(0.25),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPool2D(
    #     pool_size=(1, 1),
    #     strides=1
    # ),

    #  tf.keras.layers.Conv2D(
    #       filters=16,
    #       kernel_size=(2, 2),
    #       strides=1,
    #       padding='same',
    #       activation='relu'
    #   ),
    #     tf.keras.layers.Dropout(0.25),
    #   tf.keras.layers.BatchNormalization(),
    #   tf.keras.layers.MaxPool2D(
    #       pool_size=(2, 2),
    #       strides=2
    #   ),

    tf.keras.layers.Flatten(),  # convert 3-D to 1-D

    # tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(33, activation='softmax')
])
#print(model.summary())



sca = tf.keras.metrics.SparseCategoricalAccuracy()
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.005),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ['mae', sca, 'accuracy']
)
print('Model compiled.')

history = model.fit(X_train[..., np.newaxis], y_train, epochs = 5, batch_size = 4, validation_split = 0.1, verbose = True)


# import pickle
# file_name = 'history' 
# outfile = open(DATADIR+file_name,'wb')
# best_model = pickle.dump(history,outfile)
# outfile.close()
#
# file_name2 = 'model' 
# outfile2 = open(DATADIR+file_name2,'wb')
# best_model2 = pickle.dump(model,outfile2)
# outfile2.close()


model.evaluate(X_test[..., np.newaxis], y_test)
pred = model.predict(X_test[..., np.newaxis])


pred = np.argmax(pred, axis = 1)[:5]
label = np.argmax(y_test,axis = 1)[:5]
print(pred)
print(label)