import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import datetime
from skimage.io import imread
from skimage.transform import resize
from tensorflow.python.framework.ops import reset_default_graph

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[int(row), int(col)] = 1
    return out

# Assume all of the files from the Leaf classification Kaggle challenge is
# unpacked in the directory "Leaf/"
#
# URL: https://www.kaggle.com/c/leaf-classification/data

image_paths = glob.glob("Leaf/images/*")
print("Amount of images =", len(image_paths))

# now plot 10 images
# as we need all images to have the same dimensionality, we will resize and plot
# make the images as small as possible, until the difference between starts to get blurry
for i in range(10):
    image = imread(image_paths[i], as_grey=True)
    #image = resize(image, output_shape=(100, 100))
    plt.imshow(image, cmap='gray')
    plt.title("name: %s \n shape:%s" % (image_paths[i], image.shape))
    plt.show()

# now loading the train.csv to find features for each training point
train = pd.read_csv('Leaf/train.csv')
# notice how we "only" have 990 (989+0 elem) images for training, the rest is for testing
train.tail()

# now do similar as in train example above for test.csv
test = pd.read_csv('Leaf/test.csv')
# notice that we do not have species here, we need to predict that ..!
test.tail()

# and now do similar as in train example above for test.csv
sample_submission = pd.read_csv('Leaf/sample_submission.csv')
# accordingly to these IDs we need to provide the probability of a given plant being present
sample_submission.tail()

# name all columns in train, should be 3 different columns with 64 values each
print(train.columns[2::64])