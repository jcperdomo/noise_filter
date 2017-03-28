import numpy as np
# import the relevant model
from noise_filter.models import Image

images = np.load('images.npy')
labels = np.load('labels.npy')

NUM_IMAGES = 10000
for i in xrange(NUM_IMAGES):

     im = Image(data=images[i].tostring(), label=np.argmax(labels[i]))
     try:
         im.save()
     except:
         # if the're a problem anywhere, you wanna know about it
         print "there was a problem with image", i
