import website.wsgi
from noise_filter.models import Images

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# import the relevant model
try:
	from cPickle import dumps
except:
	from _pickle import dumps

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
images = mnist.test.images
labels = mnist.test.labels

NUM_IMAGES = len(labels)
for i in range(NUM_IMAGES):
	s = dumps(images[i])
	im = Images(data=s, label=np.argmax(labels[i]))
	im.save()
