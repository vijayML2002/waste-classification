
import pandas as pd
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
path1='D:/studies/cse/ml/image classifier/waste-classification-data/DATASET/TRAIN_1/O_1'
path2='D:/studies/cse/ml/image classifier/waste-classification-data/DATASET/TRAIN_1/R_1'
train_data=[]
train_label=[]
for img in os.listdir(path1):
    label=0
    img_array=cv2.imread(os.path.join(path1,img),cv2.IMREAD_GRAYSCALE)
    img_array=cv2.resize(img_array,(28,28))
    train_data.append(np.array(img_array))
    train_label.append(label)
    

for img in os.listdir(path2):
    label=1
    img_array=cv2.imread(os.path.join(path2,img),cv2.IMREAD_GRAYSCALE)
    img_array=cv2.resize(img_array,(28,28))
    train_data.append(np.array(img_array))
    train_label.append(label)

print('the len of the train data is :',len(train_data))
print('the len of the train label is :',len(train_label))

path1='D:/studies/cse/ml/image classifier/waste-classification-data/DATASET/TEST_1/O_1'
path2='D:/studies/cse/ml/image classifier/waste-classification-data/DATASET/TEST_1/R_1'
test_data=[]
test_label=[]
for img in os.listdir(path1):
    label=0
    img_array=cv2.imread(os.path.join(path1,img),cv2.IMREAD_GRAYSCALE)
    img_array=cv2.resize(img_array,(28,28))
    test_data.append(np.array(img_array))
    test_label.append(label)
    

for img in os.listdir(path2):
    label=1
    img_array=cv2.imread(os.path.join(path2,img),cv2.IMREAD_GRAYSCALE)
    img_array=cv2.resize(img_array,(28,28))
    test_data.append(np.array(img_array))
    test_label.append(label)

print('the length of the test data is :',len(test_data))
print('the length of the test label is :',len(test_label))


train_data=np.array(train_data)
train_new=np.array(train_data).reshape(-1,28,28,1)
train_label_new=np.array(train_label)

test_data=np.array(test_data)
test_new=np.array(test_data).reshape(-1,28,28,1)
test_label_new=np.array(test_label)


model=model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=( 28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(2,  activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_new,train_label_new,epochs=5,steps_per_epoch=10)

test_loss, test_accuracy = model.evaluate(test_new,test_label_new, steps=10)

pred=model.predict(test_new)

print(pred[4])


plt.figure()
plt.imshow(test_data[4],cmap='gray')
plt.xlabel('pred[4]')
plt.show()













