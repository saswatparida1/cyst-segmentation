import sys
import numpy
from matplotlib.image import imread
from PIL import Image
def conv(file):
    x=Image.open(file)
    x=x.convert('L')
    return x
if __name__ == '__main__':
	conv(file)
