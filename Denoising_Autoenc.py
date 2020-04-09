# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 00:06:42 2020

@author: ACER
"""
from keras.models import Model
from keras.layers import Dense,Reshape,Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Flatten
from keras.layers import MaxPooling2D,UpSampling2D
from keras import backend as K
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#input_shape
#mndata = MNIST('MNIST')
#images_train, labels_train = mndata.load_training()
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1,image_size,image_size,1])
x_test = np.reshape(x_test,[-1,image_size,image_size,1])
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

noise = np.random.normal(loc = 0.5, scale = 0.5, size = x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc = 0.5, scale = 0.5, size = x_test.shape)
x_test_noisy = x_test + noise

input_shape = (image_size,image_size,1)


inputs = Input(shape = input_shape)
x = inputs
x = Conv2D(32,(3,3),activation = 'relu')(x)
x = MaxPooling2D(pool_size = (2,2))(x)
x = Conv2D(32,(3,3),activation = 'relu')(x)
x = MaxPooling2D(pool_size = (2,2))(x)
shape = K.int_shape(x)
x = Flatten()(x)
latent = Dense(units = 16, activation = 'sigmoid')(x)

encoder = Model(inputs,latent)
encoder.summary()

latent_input = Input(shape = (16,))
x = Dense(shape[1]*shape[2]*shape[3],activation = 'sigmoid')(latent_input)
x = Reshape((shape[1],shape[2],shape[3]))(x)
x = UpSampling2D(size = (2,2))(x)
x = Conv2DTranspose(32,(3,3),activation = 'relu')(x)
x = UpSampling2D(size = (2,2))(x)
x = Conv2DTranspose(32,(3,3),activation = 'relu')(x)
output = Conv2DTranspose(1,(3,3),activation = 'relu')(x)

decoder = Model(latent_input,output)
decoder.summary()

autoencoder = Model(inputs,decoder(encoder(inputs)))
autoencoder.summary()

autoencoder.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
autoencoder.fit(x_train_noisy,x_train,validation_data=(x_test_noisy, x_test),epochs=30, batch_size=128)

x_denoised = autoencoder.predict(x_test_noisy)

rows,cols = 10,30
n = rows*cols
imgs = np.concatenate([x_test[:n],x_test_noisy[:n],x_denoised[:n]])
imgs = imgs.reshape((rows*3,cols,image_size,image_size))
imgs = np.vstack(np.split(imgs,rows,axis = 1))
imgs = imgs.reshape((rows*3,cols,image_size,image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs*255).astype(np.uint8)
plt.figure()
plt.title('Original images: top,''Noisy: Middle,''Denoised: bottom')
plt.imshow(imgs, interpolation = 'none', cmap = 'gray')
Image.fromarray(imgs).save('noisy_and_denoised.png')
plt.show()







