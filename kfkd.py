# file kfkd.py
import os

import numpy as np
import scipy as sp
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import matplotlib.pyplot as pyplot
from scipy import misc as scipyMisc
import cPickle as pickle
from numpy import linalg as LA


FTRAIN = '/Users/shkejriwal/Documents/personal/data/FacialRecognition/training.csv'
FTEST = '/Users/shkejriwal/Documents/personal/data/FacialRecognition/test.csv'


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    # print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    #     X.shape, X.min(), X.max()))
    # print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    #     y.shape, y.min(), y.max()))


    return X, y

def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y


def testingMetrics():
    train_loss = np.array([i["train_loss"] for i in net1.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


def doTest(net1):
    X, _ = load(test=True)
    y_pred = net1.predict(X)

    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred[i], ax)

    pyplot.show()



def getNeuralNetwork():
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


    net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=1000,
    verbose=1,
    )

    # X, y = load()
    # net1.fit(X, y)
    # return net1

    X, y = load2d()
    net2.fit(X, y)
    ##Training for 1000 epochs will take a while.  We'll pickle the
    ##trained model so that we can load it back later:

    with open('net.pickle', 'wb') as f:
        pickle.dump(net2, f, -1)

    return net2





def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def prepare1DImage(img):
    gray = rgb2gray(img).flatten()
    gray = np.vstack(gray) / 255.  # scale pixel values to [0, 1]
    gray = gray.astype(np.float32)
    X = np.reshape(gray,(1,-1))
    return X

def prepare2DImage(img):
    X = prepare1DImage(img)
    X = X.reshape(-1, 1, 96, 96)
    return X


def doMyTest(net1):

    #myImage = '/Users/shkejriwal/Documents/personal/data/myPics/small_no_glasses.jpg'
    #myImage = '/Users/shkejriwal/Documents/personal/data/myPics/small.jpg'
    #myImage = '/Users/shkejriwal/Documents/personal/data/myPics/small_sk_closeup.jpg'
    myImage1 = '/Users/shkejriwal/Documents/personal/data/myPics/small_full_face.jpg'
    myImage2 = '/Users/shkejriwal/Documents/personal/data/myPics/small_full_face_no_glass.jpg'

    img1 = scipyMisc.imread(myImage1)
    #X1 = prepare1DImage(img1)
    X1 = prepare2DImage(img1)

    sample1 = load(test=True)[0][6:7]

    img2 = scipyMisc.imread(myImage2)
    #X2 = prepare1DImage(img2)
    X2 = prepare2DImage(img2)
    
    y1 = net1.predict(X1)
    y2 = net1.predict(X2)

    fig = pyplot.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
    plot_sample(X1[0], y1[0],ax)
    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    plot_sample(X2[0], y2[0],ax)

    pyplot.show()



#net1 = getNeuralNetwork()
#net1 = pickle.load( open( 'net.pickle',"rb"))
#net1 = pickle.load( open( 'net2.pickle',"rb"))
#doTest(net1)
#doMyTest(net1)


# not mathematically accurate
def directedHausdorffdistance(A,B):
    points1 = np.split(A,15)
    points2 = np.split(B,15)
    print points1
    print points2
    maxVal = 0
    for a in points1:
        minVal = float("inf")
        for b in points2:
            val = np.linalg.norm(a-b)
            if val<minVal:minVal=val
        # print minVal
        val = np.linalg.norm(a-minVal)
        if val>maxVal:maxVal=val
        print maxVal
    return maxVal

            
# DO NOT USE
def HausdorffDist(A,B):

    x = directedHausdorffdistance(A,B)
    y = directedHausdorffdistance(B,A)

    return max(x,y)

    #points1 = np.split(A,15)
    #points2 = np.split(B,15)
    #return sp.spatial.distance.cdist(points1, points2 ,'euclidean')
   

 # uses a modified variant of Hausdorff Distance
 # needs to be faster and improved   
def distanceBetweenCurves(A, B):
    C1 = np.split(A,15)
    C2 = np.split(B,15)

    D = sp.spatial.distance.cdist(C1, C2, 'euclidean')

    #none symmetric Hausdorff distances
    H1 = np.max(np.min(D, axis=1))
    H2 = np.max(np.min(D, axis=0))

    return (H1 + H2) / 2.
    

def distTest(net1):

    myImage1 = '/Users/shkejriwal/Documents/personal/data/myPics/small_full_face.jpg'
    myImage2 = '/Users/shkejriwal/Documents/personal/data/myPics/small_full_face_no_glass.jpg'
    #myImage2 = myImage1

    img1 = scipyMisc.imread(myImage1)
    #X1 = prepare1DImage(img1)
    X1 = prepare2DImage(img1)

    y1 = net1.predict(X1)

    
    img2 = scipyMisc.imread(myImage2)
    #X2 = prepare1DImage(img2)
    X2 = prepare2DImage(img2)

    #X2 = load2d(test=True)[0][6:7]
    #X2 = load2d(test=True)[0][15]
    
    y2 = net1.predict(X2)

    #dist = HausdorffDist(y1[0],y2[0])
    dist = distanceBetweenCurves(y1[0],y2[0])
    print dist


    # for i in range(0,15):
    #     X2 = load2d(test=True)[0][i:i+1]
    #     #print X2
    #     y2 = net1.predict(X2)
    #     print y2
    #     dist = distanceBetweenCurves(y1[0],y2[0])
    #     print dist


#net1 = pickle.load( open( 'net.pickle',"rb"))
net1 = pickle.load( open( 'net2.pickle',"rb"))
distTest(net1)



# Takes a numpy array representaion of a grayscale image
def showImage(imgInNumpyArray):
    
    img = imgInNumpyArray.reshape(96, 96)
    imgplot = pyplot.imshow(img, cmap='gray')
    pyplot.show()


#X2 = load(test=True)[0][1]
#showImage(X2)









