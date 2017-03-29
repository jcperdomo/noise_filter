""" nnetmnist.py

The class NNetMnist trains a deep convnet on MNIST data
model from: https://www.tensorflow.org/get_started/mnist/pros

when executed as:
$ python nnetmnist.py
this script runs some usage examples

$ python nnetmnist.py --train
will train the full model on 20,000 epochs, acheiving >99% accuracy
"""

import argparse
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from os.path import join
from os.path import isfile
from os import remove

# from IPython import embed

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# TODO could have better organization using tf.variable_scope()
# see https://danijar.com/structuring-your-tensorflow-models/
class NNetMnist:
	"""
	"""
	def __init__(self, fname="mnistmodel", dirname="tmp"):
		self.graph = tf.Graph()
		fpath=join('.', dirname, fname)
		self.fname = fname
		self.dirname = dirname
		self.fpath = fpath
		self.fname_flag = "trained_flag"

		image_dim = 28
		self.image_size = image_dim*image_dim
		self.num_labels = 10
		patch_size = 5
		num_channels = 1
		depth1 = 32
		depth2 = 64
		nb_nodes = 1024
		learn_rate = 1e-4
		with self.graph.as_default():
			# inputs
			self.x = tf.placeholder(tf.float32, shape=[None, self.image_size])
			self.y_ = tf.placeholder(tf.float32, shape=[None, self.num_labels])
			self.keep_prob = tf.placeholder(tf.float32)
			# variables
			W_conv1 = self.__weight_variable([patch_size, patch_size, num_channels, depth1])
			b_conv1 = self.__bias_variable([depth1])
			W_conv2 = self.__weight_variable([patch_size, patch_size, depth1, depth2])
			b_conv2 = self.__bias_variable([depth2])
			W_fc1 = self.__weight_variable([image_dim // 4 * image_dim // 4 * depth2, nb_nodes])
			b_fc1 = self.__bias_variable([nb_nodes])

			def model(X):
				x_image = tf.reshape(X, [-1,image_dim,image_dim,num_channels])
				# first convolutional and max pooling layer
				h_conv1 = tf.nn.relu(self.__conv2d(x_image, W_conv1) + b_conv1)
				h_pool1 = self.__max_pool_2x2(h_conv1)
				# second convolutional and max pooling layer
				h_conv2 = tf.nn.relu(self.__conv2d(h_pool1, W_conv2) + b_conv2)
				h_pool2 = self.__max_pool_2x2(h_conv2)
				# densely connected layer
				h_pool2_flat = tf.reshape(h_pool2, [-1, image_dim // 4 * image_dim // 4 * depth2])
				h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
				# dropout layer
				h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
				# readout layer
				W_fc2 = self.__weight_variable([nb_nodes, self.num_labels])
				b_fc2 = self.__bias_variable([self.num_labels])
				return tf.matmul(h_fc1_drop, W_fc2) + b_fc2

			self.logits = model(self.x)
			self.loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits))

			# gradient of loss with respect to input
			self.grad_input = tf.gradients(self.loss, [self.x])[0]

			# training
			self.train_step = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)
			self.correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.y_,1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

			self.__predict = tf.nn.softmax(self.logits)

			self.init_op = tf.global_variables_initializer()
			self.saver = tf.train.Saver()

	def __weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def __bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def __conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def __max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
			strides=[1, 2, 2, 1], padding='SAME')

	def fit(self, epochs=20000, batch_size=32, dropout_rate=0.5, verbose=True,
		save_model=True, force_retrain=False, skip_if_trained=False):
		""" trains on all samples in TF defined training set
		param epochs := number of batches to train on
		param batch_size := number of samples in each update
		param dropout_rate := probability that any node in the dropout later is dropped
		returns 0 if successfully trained
		"""
		# hacky way to make sure to not train if a good model is already trained
		fpath_flag = join('.', self.dirname, self.fname_flag)
		if skip_if_trained and not force_retrain:
			if isfile(fpath_flag):
				return 0
			# create file flag if not exist
			with open(fpath_flag, 'w') as f:
				pass

		verboseprint = print if verbose else lambda *a, **k: None

		with tf.Session(graph=self.graph) as session:
			# TODO if there exists a saved model, initialize from there
			if not force_retrain:
				verboseprint("Restoring model.")
				self.saver.restore(session, self.fpath)
			else:
				session.run(self.init_op)
				if not skip_if_trained:
					# removed trained flag only if restarted from scratch and skip if trained is false
					remove(join('.', self.dirname, self.fname_flag))
			verboseprint("Training...")

			for i in range(epochs):
				batch = mnist.train.next_batch(batch_size)
				if i%1000 == 0:
					train_accuracy = self.accuracy.eval(feed_dict={self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
					verboseprint("step {}, training accuracy {}".format(i, train_accuracy))
				self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: dropout_rate})

			test_acc = self.accuracy.eval(feed_dict=
				{self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0})
			verboseprint("test accuracy {}".format(test_acc))

			# save the model to disk
			if save_model:
				verboseprint("Saving model...")
				p = self.saver.save(session, self.fpath)
				verboseprint("Saved model to {}".format(p))

		return 0

	# TODO this is very similar to gradient, organize better
	def predict(self, X):
		""" returns predictions on input X using model previously saved on disk
		the predictions vectors of length numsamples that are normalized to 1
		"""
		if len(X.shape) != 2 or X.shape[1] != self.image_size:
			raise IndexError("Shape of input data must be (n, {}), where n is the number of samples.".format(self.image_size))

		preds = None
		with tf.Session(graph=self.graph) as session:
			try:
				self.saver.restore(session, self.fpath)
			except:
				raise ValueError("Could not restore model. Has it been saved to directory: {}?".format(self.dirname))

			preds = self.__predict.eval(feed_dict={self.x:X, self.keep_prob: 1.0})

		return preds

	def gradient(self, X, Y):
		""" returns gradient of loss with respect to inputs X
		return value will have the same shape as X
		"""
		if len(X.shape) != 2 or X.shape[1] != self.image_size:
			raise IndexError("Shape of input data must be (n, {}), where n is the number of samples.".format(self.image_size))
		if len(Y.shape) != 2 or Y.shape[1] != self.num_labels:
			raise IndexError("Shape of input data must be (n, {}), where n is the number of labels.".format(self.num_labels))

		grads = None
		with tf.Session(graph=self.graph) as session:
			try:
				self.saver.restore(session, self.fpath)
			except:
				raise ValueError("Could not restore model. Has it been saved to directory: {}?".format(self.dirname))

			grads = self.grad_input.eval(feed_dict={self.x:X, self.y_:Y, self.keep_prob: 1.0})

		return grads


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--train", help="train model for 20,000 epochs and overwrite previous")
	args = parser.parse_args()

	if args.train:
		nn = NNetMnist()
		nn.fit(skip_if_trained=True, force_retrain=True, verbose=False)
	else:
		nn = NNetMnist()
		# model is saved by default
		nn.fit(epochs=101)
		# this one is retrained from scratch
		nn.fit(epochs=101, force_retrain=True)
		# previous model is loaded by default , so this improves on previous parameters
		nn.fit(epochs=101)
		# skip_if_trained sets a flag that tells future calls to fit to not do any work, unless force_retrain is set
		nn.fit(epochs=201, skip_if_trained=True, force_retrain=True)
		# this one will not do any work

		nn.fit(epochs=201, skip_if_trained=True)

		# predictions are returned as probability distribution vectors
		p = nn.predict(mnist.test.images)
		print(np.sum(p, axis=1))
		# gradients with respect to input
		g = nn.gradients(mnist.test.images, mnist.test.labels)
		print(g)
