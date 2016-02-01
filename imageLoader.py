import numpy as np
import matplotlib.pyplot as pyplot
from scipy import misc

from PIL import Image


myImage = '/Users/shkejriwal/Documents/personal/data/myPics/small.jpg'


#image = Image.open(myImage).convert('LA')
#image = misc.imread(myImage)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = misc.imread(myImage)     
image = rgb2gray(img)

print image

pyplot.imshow(image, cmap = pyplot.get_cmap('gray'))
pyplot.show()
