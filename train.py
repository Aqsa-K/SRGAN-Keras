import tensorflow as tf
import numpy as np
from tqdm import tqdm
from VGG_LOSS import VGG_MODEL
import utils
import networks
import os



np.random.seed(10)
#downscale factor for creating Low Resolution images for training the generator
downscale_factor = 4
image_shape = (148,148,3)
train_directory = 'data/train/combined_data/'


model_save_dir = './checkpoints/saved_models/'
if not os.path.exists('./checkpoints'):
  os.makedirs('./checkpoints')

if not os.path.exists(model_save_dir):
  os.makedirs(model_save_dir)

checkpoint_path = "checkpoints/training.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback - save weights after every 10 epochs 
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=10)


def train(epochs, batch_size, input_dir, model_save_dir):
  

  # Make an instance of the VGG class
  vgg_model = VGG_MODEL(image_shape) 

  # Get High-Resolution(HR) [148,148,3] in this case and corresponding Low-Resolution(LR) images
  x_train_lr, x_train_hr  = utils.load_training_data(input_dir, [148,148,3])
  
  #Based on the the batch size, get the total number of batches
  batch_count = int(x_train_lr.shape[0]/batch_size)
  
  #Get the downscaled image shape based on the downscale factor
  image_shape_downscaled = utils.get_downscaled_shape(image_shape, downscale_factor)

  # Initialize the generator network with the input image shape as the downscaled image shape (shape of LR images)
  generator = networks.Generator(input_shape=image_shape_downscaled)
  
  # Initialize the discriminator with the input image shape as the original image shape (HR image shape)
  discriminator = networks.Discriminator()
  
  # Get the optimizer to tweak parameters based on loss
  optimizer = vgg_model.get_optimizer()
  
  # Compile the three models - generator, discriminator and gan(comb of both gen and disc - this network will train generator and will not tweak discriminator)
  generator.compile(loss=vgg_model.vgg_loss, optimizer=optimizer)
  discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
  gan = networks.GAN_Network(generator, discriminator, image_shape_downscaled, optimizer, vgg_model.vgg_loss)
  
  
  # Run training for the number of epochs defined
  for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            
            # Get the next batch of LR and HR images
            image_batch_lr, image_batch_hr = utils.get_random_batch(x_train_lr, x_train_hr, x_train_hr.shape[0], batch_size)

            generated_images_sr = generator.predict(image_batch_lr)
            print(generated_images_sr.shape)

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data_Y = np.random.random_sample(batch_size)*0.2
            
            discriminator.trainable = True
            print(real_data_Y.shape)
            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
            
            
        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        gan_loss = str(gan_loss)
        
        if e % 1 == 0:
            generator.save_weights(model_save_dir + 'gen_model%d.h5' % e)
            discriminator.save_weights(model_save_dir + 'dis_model%d.h5' % e)
  
  networks.save_model(gan)


cwd = os.getcwd()
print("working directory", cwd)
train(100, 1, train_directory, model_save_dir)
