import os

import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg



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






