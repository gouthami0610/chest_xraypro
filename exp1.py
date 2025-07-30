import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report,confusion_matrix #used for report
from tensorflow.keras.models import Sequential  # image into pixecl(sequential is the model)
from tensorflow.keras.layers import Dense,Flatten  
import numpy as np
#making all the pics to same size
train_datagen=ImageDataGenerator(rescale=1./255,zoom_range=0.2,horizontal_flip=True)
train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
train = train_datagen.flow_from_directory(
    "archive/chest_xray/chest_xray/train",target_size=(150,150),
                                          batch_size=32, #takes 32 images at a time
                                          class_mode='binary') #convert to binary

test = test_datagen.flow_from_directory(
    "archive/chest_xray/chest_xray/test",target_size=(150,150),
                                          batch_size=32, #takes 32 images at a time
                                          class_mode='binary') #convert to binary
model=Sequential([tf.keras.layers.Conv2D(32,(3,3),
                                        activation='relu',
                                        input_shape=(150,150,3)), 
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        tf.keras.layers.Conv2D(64,(3,3),
                                        activation='relu'),tf.keras.layers.MaxPooling2D(2,2),
                                         tf.keras.layers.Flatten(),
                                          tf.keras.layers.Dense(128,activation='relu'),
                                           tf.keras.layers.Dense(1,activation='sigmoid')
                                        ])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train,validation_data=test,epochs=5)
model.summary()
model.save("pneumonia_model.h5")
#32 filters of matrix size 3*3
#input_shape =150*150 ima into 3 col RGB

