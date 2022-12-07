import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import random
import torchvision

from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from torchsummary import summary
from PIL import Image

class ResNet50():

    def __init__(self):
        self.img = None
        self.trainPath = ''
        self.validPath = ''
        self.image_size = (224, 224)
        self.batch_size = 32
        self.train_ds = None
        self.val_ds = None
        self.model = None
        self.result = None
        pass

    def LoadImage(self,fileName):
        # at home
        '''self.trainPath = train
        self.validPath = valid

        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train,
        image_size=self.image_size,
        batch_size=self.batch_size,)

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid,
        image_size=self.image_size,
        batch_size=self.batch_size)

        self.train_ds = self.train_ds.prefetch(buffer_size=32)
        self.val_ds = self.val_ds.prefetch(buffer_size=32)'''

        # demo 
        self.img = fileName

    def ShowImages(self):
        catImg = cv2.imread('inference_dataset\Cat\8043.jpg')
        dogImg = cv2.imread('inference_dataset\Dog\medium_2022-09-20-a968b7b738.jpg')

        catImg = cv2.cvtColor(catImg, cv2.COLOR_BGR2RGB)
        dogImg = cv2.cvtColor(dogImg, cv2.COLOR_BGR2RGB)

        catImg = cv2.resize(catImg, (224,224))
        dogImg = cv2.resize(dogImg, (224,224))

        fig = plt.figure(figsize=(1, 2))
        
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.title('Cat')
        plt.imshow(catImg)

        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.title('Dog')
        plt.imshow(dogImg)

        plt.show()



    def ShowDistribution(self):

        # at home
        '''catDiR = self.trainPath + '/Cat'
        dogDiR = self.trainPath + '/Dog'

        catCnt = len([name for name in os.listdir(catDiR) if os.path.isfile(os.path.join(catDiR, name))])
        dogCnt = len([name for name in os.listdir(dogDiR) if os.path.isfile(os.path.join(dogDiR, name))])

        x = ['Cat','Dog']
        h = [catCnt, dogCnt]

        for a,b in zip(x,h):
            plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)

        plt.bar(x,h)
        plt.ylabel('Number of images')
        plt.title('Class Distribution')
        plt.savefig('ClassDistribution.png')
        plt.show()'''

        # demo
        img = cv2.imread('ClassDistribution.png')
        cv2.imshow('Class Distribution ', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def ShowModelStructure(self):
        # at home and demo
        self.model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), classes=2)
        model = tf.keras.models.Sequential()
        model.add(self.model)
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        self.model = model
        print(self.model.summary())

        # when training
        # self.Training()

    def ShowComparison(self):
        # at home
        '''model1 = tf.keras.models.load_model('myModel1.h5')
        model2 = tf.keras.models.load_model('myModel2.h5')

        score1 = model1.evaluate(self.val_ds, verbose=0)
        score2 = model2.evaluate(self.val_ds, verbose=0)

        x = ['Binary Cross Entropy','Focal Loss']
        h = [score2[1]*100, score1[1]*100]

        for a,b in zip(x,h):
            plt.text(a, b+0.01, str(b), ha='center', va= 'bottom',fontsize=7)

        plt.bar(x,h)
        plt.ylabel('Accuracy(%)')
        plt.title('Accuracy Comparison')
        plt.savefig('AccuracyComparison.png')
        plt.show()'''

        # demo
        img = cv2.imread('AccuracyComparison.png')
        cv2.imshow('Accuracy Comparison', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Inference(self):
        if self.img != None:
            model = tf.keras.models.load_model('myModel1.h5')
            img = Image.open(self.img)
            x = np.array(img, dtype='float32')
            x = np.expand_dims(x, axis=0)
            pred = model.predict(x)

            if pred[0] > 0.5:
                c = 'Dog'
            else:
                c = 'Cat'
            return (c)
        return ''

    def Training(self):

        #1st
        loss_function = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.4, gamma=1.0)

        #2st
        loss_function = tf.keras.losses.BinaryCrossentropy()

        adam = tf.keras.optimizers.Adam(learning_rate=8e-5)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

        self.model.compile(optimizer=adam,
            loss=loss_function,
            metrics=['accuracy'])

        self.result = self.model.fit(self.train_ds, epochs=3, callbacks=[callback], verbose = 1)

        self.model.save('myModel2.h5')



    
