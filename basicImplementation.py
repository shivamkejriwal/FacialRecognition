import os

import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet




FTRAIN = '/Users/shkejriwal/Documents/personal/data/FacialRecognition/training.csv'
FTEST = '/Users/shkejriwal/Documents/personal/data/FacialRecognition/test.csv'


# Takes a numpy array representaion of a grayscale image
def showImage(imgInNumpyArray):
	
	img = imgInNumpyArray.reshape(96, 96)
	imgplot = plt.imshow(img, cmap='gray')
	plt.show()



def loadTrainingSet():

	print "Loading data ...."
	df = read_csv(os.path.expanduser(FTRAIN))  # load pandas dataframe

	print "Converting image data to numpy array .."
	df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

	print " Lodading Done."
	#oneImage = df['Image'][0]
	#showImage(oneImage)

	print "Preprocessing dataset .."
	#print(df.count())  # prints the number of values for each column
	df = df.dropna()  # drop all rows that have missing values in them

	#print "Dropped all rows with missing values"
	#print(df.count())

	X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
	X = X.astype(np.float32)

	#print df.columns[:-1] # gives a list of all headers except for the last
	y = df[df.columns[:-1]].values
	y = (y - 48) / 48  # scale target coordinates to [-1, 1].
	#X, y = shuffle(X, y, random_state=42)  # shuffle train data
	y = y.astype(np.float32)

	return X, y


X,y = loadTrainingSet()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))
print X[:1]
print y[:1]


def NN(X,y):

	net1 = NeuralNet(
	    layers=[  # three layers: one hidden layer
	        ('input', layers.InputLayer),
	        ('hidden', layers.DenseLayer),
	        ('output', layers.DenseLayer),
	        ],
	    # layer parameters:
	    input_shape=(None, 9216),  # 96x96 input pixels per batch
	    hidden_num_units=100,  # number of units in hidden layer
	    output_nonlinearity=None,  # output layer uses identity function
	    output_num_units=30,  # 30 target values

	    # optimization method:
	    update=nesterov_momentum,
	    update_learning_rate=0.01,
	    update_momentum=0.9,

	    regression=True,  # flag to indicate we're dealing with regression problem
	    max_epochs=400,  # we want to train this many epochs
	    verbose=1,
	    )

	net1.fit(X, y)

NN(X,y)



def singleImage():







