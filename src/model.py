import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os,cv2
import warnings
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend  as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dense,Dropout,Rescaling,Dense,Flatten,Activation,BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models

for dirname, _, filenames in os.walk('../YogaDataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

warnings.filterwarnings("ignore")

data_path = '../YogaDataset'

labels=[]
for folder in os.listdir(data_path):
    labels.append(folder)
labels.sort() #len = 107
labels = labels[0:13] #len =13
print(labels)

train_images=[]
train_labels=[]

for i,folder in enumerate(labels):
    try:
        for image in os.listdir(data_path+'/'+folder):
            img = os.path.join(data_path+'/'+folder+'/'+image)
            img = cv2.imread(img)
            img = cv2.resize(img,(256,256))
            train_images.append(img)
            train_labels.append(i)
    except:
        print(i,folder,image,img)
train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels).astype('int64')

for i in [300,400]:
    plt.imshow(train_images[i])
    plt.title(labels[train_labels[i]])
    #plt.show()

train_labels = to_categorical(train_labels, 13)
print(f'After preprocessing, our dataset has {train_images.shape[0]} images with shape {train_images.shape[1:]}')
print(f'After preprocessing, our dataset has {train_labels.shape[0]} rows with {train_labels.shape[1]} labels')

X_train,X_test,y_train,y_test = train_test_split(train_images,train_labels,test_size=0.1,shuffle=True)

print(f'After spiltting, shape of our train dataset: {X_train.shape}')
print(f'After spiltting, shape of our test dataset: {X_test.shape}')

K.clear_session()
model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(256,256,3)),
            
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            Rescaling(1.0 / 255),
            
            Conv2D(32,(3,3),activation='relu'),
            MaxPooling2D((2,2)),  
            Dropout(0.3),

            Conv2D(64,(3,3),activation='relu'),
            MaxPooling2D((2,2)),    
            Dropout(0.3),
            
            Conv2D(64,(3,3),activation='relu'),
            MaxPooling2D((2,2)),
            Dropout(0.5),
            
            Flatten(),
            Dense(512,activation='relu'),
            Dense(128,activation='relu'),    
            Dense(13,activation='softmax')
        
])
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
            )

history_model = model.fit(X_train, y_train,
          batch_size=32, epochs=20, validation_split=0.2)

plt.plot(history_model.history['accuracy'])
plt.plot(history_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history_model.history['loss'])
plt.plot(history_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


test_img = cv2.imread('../YogaDataset/ananda balasana/10-0.png')
test_img = cv2.resize(test_img,(256,256))
test_img1 = np.asarray(test_img)
test_img = test_img1.reshape(-1,256,256,3)
p = model.predict(test_img)

plt.imshow(test_img1)
plt.title(labels[np.argmax(p)])
np.argmax(p)

