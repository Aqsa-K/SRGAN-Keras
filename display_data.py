%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def display_data(TRAINING_DIR, TESTING_DIR):
    train_dir_names = os.listdir(TRAINING_DIR)
    test_dir_names = os.listdir(TESTING_DIR)

    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4

    # Index for iterating over images
    pic_index = 0

    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8
    next_train_pix = [os.path.join(TRAINING_DIR, fname) 
                    for fname in train_dir_names[pic_index-8:pic_index]]
    next_test_pix = [os.path.join(TESTING_DIR, fname) 
                    for fname in test_dir_names[pic_index-8:pic_index]]


    for i, img_path in enumerate(next_train_pix+next_test_pix):
      # Set up subplot; subplot indices start at 1
      sp = plt.subplot(nrows, ncols, i + 1)
      sp.axis('Off') # Don't show axes (or gridlines)

      img = mpimg.imread(img_path)
      print(img.shape)
      plt.imshow(img)

    plt.show()