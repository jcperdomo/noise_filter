import website.wsgi
from noise_filter.models import Images

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# import the relevant model
try:
	from cPickle import dumps
except:
	from _pickle import dumps

from IPython import embed

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
images = mnist.test.images
labels = mnist.test.labels

# clear db first
Images.objects.all().delete()

NUM_IMAGES = len(labels)
for i in range(NUM_IMAGES):
	s = dumps(images[i]).decode("latin-1")
	im = Images(data=s, label=np.argmax(labels[i]))
	im.save()
