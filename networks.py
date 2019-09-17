import numpy as np
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D

from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model

def identity_block(X, f, filters, stage, block):
  """
  Implementation of the identity block
  
  Arguments:
  X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
  f -- integer, specifying the shape of the middle CONV's window for the main path
  filters -- python list of integers, defining the number of filters in the CONV layers of the main path
  stage -- integer, used to name the layers, depending on their position in the network
  block -- string/character, used to name the layers, depending on their position in the network
  
  Returns:
  X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
  """
  
  # defining name basis
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  
  # Retrieve Filters
  F1, F2 = filters
  
  # Save the input value. You'll need this later to add back to the main path. 
  X_shortcut = X
  
  # First component of main path
  X = Conv2D(filters = F1, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2a')(X)
  X = BatchNormalization(momentum=0.5, name = bn_name_base + '2a')(X)
  X = Activation('relu')(X)
  
  # Second component of main path
  X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b')(X)
  X = BatchNormalization(momentum=0.5, name = bn_name_base + '2b')(X)


  # Final step: Add shortcut value to main path, and pass it through a RELU activation
  X = Add()([X_shortcut, X])
  
  return X


def up_block(X, f, filter_size, stage, block):
  
  # defining name basis
  conv_name_base = 'srg' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  
  # perform upsampling
  X = Conv2D(filters = filter_size, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b')(X)
  X = UpSampling2D(size=2)(X)
  X = Activation('relu')(X)
  
  return X


def Generator(input_shape):
  """
  Implementation of the popular SRGAN the following architecture:
  CONV2D -> RELU -> identity_block -> identity_block -> identity_block -> identity_block -> 
  CONV2D -> RELU -> ADD -> UPBLOCK -> UPBLOCK -> CONV2D -> TANH -> to Discriminator
  Arguments:
  input_shape -- shape of the images of the dataset
  Returns:
  model -- a Model() instance in Keras
  """
  
  # Define the input as a tensor with shape input_shape
  X_input = Input(input_shape)

  # Zero-Padding
  X = Conv2D(filters = 64, kernel_size = (9, 9), strides = (1, 1), padding = 'same')(X_input)
  X = Activation('relu')(X)
  
  X_shortcut = X
  
  X = identity_block(X, 3, filters=[64,64], stage=2, block='a')
  X = identity_block(X, 3, filters=[64,64], stage=2, block='b')
  X = identity_block(X, 3, filters=[64,64], stage=2, block='c')
  X = identity_block(X, 3, filters=[64,64], stage=2, block='e')
  X = identity_block(X, 3, filters=[64,64], stage=2, block='f')
  
  X = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(X)
  # X = Activation('relu')(X)
  
  X = Add()([X_shortcut, X])
  
  
  X = up_block(X, 3, 256, stage=3, block='a')
  X = up_block(X, 3, 256, stage=3, block='b')

  X = Conv2D(filters=3, kernel_size=(9,9), strides=(1,1), padding='same')(X)
  X = Activation('tanh')(X)
  
  generator_model = Model(inputs=X_input, outputs=X)
  
  return generator_model

  
def ConvBlock(x, filters, kernel_size, strides):
  x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(x)
  x = BatchNormalization(momentum=0.5)(x)
  x = LeakyReLU(alpha=0.2)(x)
  return x

def Discriminator(input_shape = (148, 148, 3)):
  """
  Implementation of the popular SRGAN the following architecture:
  CONV2D -> LEAKY RELU -> 
  CONV2D -> CONV2D -> CONV2D -> CONV2D -> CONV2D -> CONV2D -> CONV2D ->
  DENSE -> LEAKY RELU-> DENSE -> SIGMOID
  
  Arguments:
  input_shape -- shape of the images of the dataset
  Returns:
  model -- a Model() instance in Keras
  """
  
  # Define the input as a tensor with shape input_shape
  X_input = Input(input_shape)

  # Zero-Padding
  X = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(X_input)
  X = LeakyReLU(alpha=0.2)(X)
  
  X = ConvBlock(X, 64, 3, 2)
  X = ConvBlock(X, 128, 3, 1)
  X = ConvBlock(X, 128, 3, 2)
  X = ConvBlock(X, 256, 3, 1)
  X = ConvBlock(X, 256, 3, 2)
  X = ConvBlock(X, 512, 3, 1)
  X = ConvBlock(X, 512, 3, 2)

  X = Flatten()(X)
  X = Dense(1024)(X)
  X = LeakyReLU(alpha=0.2)(X)
  
  X = Dense(1)(X)
  X = Activation('sigmoid')(X)
  
  discriminator_model = Model(inputs=X_input, outputs=X)
  
  return discriminator_model


def GAN_Network(generator, discriminator, image_shape, optimizer, vgg_loss):
  discriminator.trainable = False
  gan_input = Input(shape=image_shape)
  generator_output = generator(gan_input)
  discriminator_output = discriminator(generator_output)
  gan = Model(inputs=gan_input, outputs=[generator_output, discriminator_output])
  
  gan.compile(loss=[vgg_loss, "binary_crossentropy"], 
              loss_weights = [1., 1e-3],
             optimizer=optimizer)
  
  return gan

def save_model(model):
  # Save the weights
  model.save_weights('checkpoints/celeb_faces_weights.ckpt')
  # Save entire model to a HDF5 file
  model.save('checkpoints/celeb_faces_model.h5')
    
