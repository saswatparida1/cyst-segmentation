import sys
import numpy as np
import matplotlib
from matplotlib import image
from PIL import Image
x=np.load("sas.npy")
x=np.asarray(x)
matplotlib.image.imsave('input.png',x)
x=Image.open("/home/saswat/PycharmProjects/saswat/input.png")
y=Image.open("/home/saswat/PycharmProjects/saswat/name1.png")
z=Image.blend(x,y,0.5)
matplotlib.image.imsave('overlay.png',z)
