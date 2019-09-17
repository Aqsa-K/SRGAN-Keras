from skimage import data, io, filters
import scipy
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.image as mpimg
import os
import numpy as np
import matplotlib.pyplot as plt

# Takes list of images and provide HR images in form of numpy array
def hr_images(images):
    images_hr = np.array(images, dtype='float32')
    return images_hr
  
# Takes list of images and provide LR images in form of numpy array
def lr_images(images_real , downscale):
    
    images = []
    for img in  range(len(images_real)):
        images.append((resize(images_real[img], [images_real[img].shape[0]//downscale,images_real[img].shape[1]//downscale]) * 255).astype(np.uint8))
    images_lr = np.array(images)
    return images_lr
    
def normalize(input_data):
    x = input_data.astype(np.float32)
    x_norm = (x-127.5)/127.5
    return x_norm
  
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data
    
def load_training_data(directory, hr_image_shape):

    #read images into x_train 
    x_train = []
    
    for file_name in os.listdir(directory):
      img_path = os.path.join(directory, file_name)
      img = data.imread(img_path)
      img = resize(img, hr_image_shape , anti_aliasing=True)
      x_train.append(img)
      
    x_train_hr = hr_images(x_train)
    x_train_hr = normalize(x_train_hr)
    
    x_train_lr = lr_images(x_train, 4)
    x_train_lr = normalize(x_train_lr)

    
    return x_train_lr, x_train_hr

def get_downscaled_shape(image_shape, downscale_factor):
    downscaled_shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])
    return downscaled_shape
    

def get_random_batch(x_train_lr, x_train_hr, num_images, batch_size):
    rand_nums = np.random.randint(0, num_images, size=batch_size)
    print("rand_nums: ", rand_nums)
    
    image_batch_hr = x_train_hr[rand_nums]
    print(image_batch_hr.shape)
    image_batch_lr = x_train_lr[rand_nums] 

    return image_batch_lr, image_batch_hr
    