import numpy as np
# import the relevant model
from noise_filter.models import Images
from cPickle import dumps

images = np.load('images.npy')
labels = np.load('labels.npy')

NUM_IMAGES = 10000
for i in xrange(NUM_IMAGES):
    s = dumps(images[i])
    im = Image(data=s, label=np.argmax(labels[i]))
    im.save()
