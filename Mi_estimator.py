#!/usr/bin/env python

"""regularizers.py: Where the MI estimators (also used as regularizers) are defined."""

__author__ = "Malik Boudiaf"
__version__ = "0.1"
__maintainer__ = "Malik Boudiaf"
__email__ = "malik-abdelkrim.boudiaf.1@ens.etsmtl.ca"
__status__ = "Development"

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
from ops import *
from tqdm import trange
from critic_architectures import joint_critic, separate_critic

class Mi_estimator(object):

	def __init__(self, regu_name, dim_x, dim_z, batch_size, critic_layers=[256, 256, 256], critic_lr=1e-4, critic_activation='relu', critic_type='joint', ema_decay=0.99, negative_samples=1):
		"""
		Description
		-----------
		Defines the mi_estimator

		Parameters
		----------
		regu_name: string
			Name of the regularization method used. Chosen in ['nce','mine','nwj']
		dim_x: list
			Dimension of X (all but batch dimension)
		dim_z: list
			Dimension of Z (all but batch dimension)
		batch_size: int
			Size of the batches that will be used during training.
		critic_layers: list
			The fully connected layers used in the critic network.
		critic_lr: float
			Learning rate used to update the critic network
		critic_activation : string
			Activation used in the critic network
		critic_type: string
			Type of critic network used. Between 'joint' and 'separate'
			Joint critics usually work better, but separate critics are more computationally efficient
		ema_decay: float
			Exponential Moving Average decay used to correct biased gradients (for MINE only)
		negative_samples: int
			Number of negative samples used in the expectation estimation.
			Number between 1 and batch_size - 1
			The more, the better but the more computationally expensive.

		Returns
		-------
		mi_estimator: Mi_estimator
			The mutual information estimator
		"""
		self.regu_name = regu_name
		self.critic_lr = critic_lr
		self.ema_decay = ema_decay
		self.batch_size = batch_size
		self.critic_activation = critic_activation
		self.critic_layers = critic_layers
		self.ema_decay = ema_decay
		self.dim_x = dim_x
		self.dim_z = dim_z
		if critic_type == 'separate':
			self.negative_samples = self.batch_size # if critic is separate, we get 'for free' all the n^2 combinations
		else:
			self.negative_samples = negative_samples 

		self.critic = eval('{}_critic(dim_x, dim_z, critic_activation, critic_layers, negative_samples)'\
						   .format(critic_type, dim_x, dim_z, critic_activation, critic_layers, negative_samples))

	def nwj(self, x, z):
		"""
		Description
		-----------
		An implementation of the f-divergence based MI estimator (NWJ)
		https://arxiv.org/abs/1606.00709

		Parameters
		----------
		x : tf.Variable [batch_size, dim_x]
			Representing a batch of samples from P_X
		z : tf.Variable [batch_size, dim_z]
			Representing a batch of samples from P_Z|X=x

		Returns
		-------
		mi : tf.Variable []
			An estimate of the MI between x and z
		mi_for_grads : tf.Variable []
			A bias corrected version of mi between x and z.
			Must only be used to get a better gradient estimate of the MI.
		"""
		T_joint, T_product = self.critic(x, z)
		E_joint = 1 / self.batch_size * tf.reduce_sum(T_joint)
		E_product = 1 / (np.e * self.batch_size * self.negative_samples) * (tf.reduce_sum(tf.exp(T_product)) - self.batch_size)
		mi = E_joint - E_product
		mi_for_grads = mi

		return mi, mi_for_grads

	def mine(self, x, z):
		"""
		Description
		-----------
		An implementation of the Donsker Varadhan based MI estimator (MINE)
		https://arxiv.org/abs/1801.04062

		Parameters
		----------
		x : tf.Variable [batch_size, dim_x]
			Representing a batch of samples from P_X
		z : tf.Variable [batch_size, dim_z]
			Representing a batch of samples from P_Z|X=x

		Returns
		-------
		mi : tf.Variable []
			An estimate of the MI between x and z
		mi_for_grads : tf.Variable []
			A bias corrected version of mi between x and z.
			Must only be used to get a better gradient estimate of the MI.
		"""
		T_joint, T_product = self.critic(x, z)

		E_joint = 1 / self.batch_size * tf.reduce_sum(T_joint)
		E_product = np.log(1 / (self.batch_size * self.negative_samples)) + tf.log(tf.reduce_sum(tf.exp(T_product)) - self.batch_size)
		mi = E_joint - E_product

		ema_denominator = tf.Variable(tf.exp(tf.reduce_logsumexp(T_product)))
		ema_denominator -= (1 - self.ema_decay) * (ema_denominator - tf.exp(tf.reduce_logsumexp(T_product)))
		mi_for_grads = E_joint - 1 / tf.stop_gradient(ema_denominator) * tf.exp(tf.reduce_logsumexp(T_product))

		return mi, mi_for_grads

	def nce(self, x, z):
		"""
		Description
		-----------
		An implementation of the noise-contrastive based MI estimator (NWJ)
		https://arxiv.org/abs/1807.03748

		Parameters
		----------
		x : tf.Variable [batch_size, dim_x]
			Representing a batch of samples from P_X
		z : tf.Variable [batch_size, dim_z]
			Representing a batch of samples from P_Z|X=x

		Returns
		-------
		mi : tf.Variable []
			An estimate of the MI between x and z
		mi_for_grads : tf.Variable []
			A bias corrected version of mi between x and z.
			Must only be used to get a better gradient estimate of the MI.
		"""

		T_joint, T_product = self.critic(x, z)
		E_joint = 1 / self.batch_size * tf.reduce_sum(T_joint)
		E_product = np.log(1 / (self.negative_samples+1)) + tf.reduce_mean(tf.reduce_logsumexp(tf.add(T_joint, T_product), axis=1))

		mi = E_joint - E_product
		mi_for_grads = mi

		return mi, mi_for_grads

	@property
	def vars(self):
		vars = {}
		vars['critic'] = [var for var in tf.global_variables() if 'critic' in var.name]
		return vars

	def __call__(self, x, z, optimizer=None):
		"""
		Description
		-----------
		Method to call whenever MI is to be used as a regularization term in another loss

		Parameters
		----------
		x : tf.Variable [batch_size, dim_x]
			Representing a batch of samples from P_X
		z : tf.Variable [batch_size, dim_z]
			Representing a batch of samples from P_Z|X=x

		Returns
		-------
		train_ops : list
			The list of tensorflow operations to perform in order to update the bound.
		quantities : dict
			Dictionnary {'mi': current estimate of MI,
						 'mi_for_grads' : bias corrected estimate of MI, term to be used for differentiation}
		"""

		train_ops = {}
		quantities = {}

		mi, mi_for_grads = eval("self.{}(x,z)".format(self.regu_name))

		quantities['mi'] = mi
		quantities['mi_for_grads'] = mi_for_grads
		if optimizer is None:
			optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
		train_ops['critic'] = optimizer.minimize(- mi_for_grads, var_list=self.vars['critic'])

		return train_ops, quantities

	def get_info(self, sess, feed_dict, quantities, epoch, step):
		info = []
		info.append("Epoch={:d} Step={:d}".format(epoch, step))
		values = sess.run(list(quantities.values()), feed_dict=feed_dict)
		for name, value in zip(quantities.keys(), values):
			info.append("{}={:.3g}".format(name, value))
		return info


	def fit(self, x_data, z_data, epochs, ground_truth_mi=None):
		"""
		Description
		-----------
		Method to call whenever one only need a scalar estimate of I(x_data, z_data)

		Parameters
		----------
		x : numpy array [?, dim_x]
			Representing a dataset of samples from P_X
		z : numpy array [?, dim_z]
			Representing a dataset of samples from P_Z|X=x

		Returns
		-------
		mi_eval : float
			An estimate of I(x_data, z_data)
		"""

		tf.reset_default_graph()
		x_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size] + list(x_data.shape[1:]))
		z_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size] + list(z_data.shape[1:]))

		train_ops, quantities = self(x_ph, z_ph)

		gpu_options = tf.GPUOptions(allow_growth=True)
		
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			n_batchs = int(x_data.shape[0] / self.batch_size)
			epoch_bar = trange(epochs)
			mi_ema = 0
			for epoch in epoch_bar:
				batch_bar = trange(n_batchs)
				for i in batch_bar:
					x_batch = x_data[i * self.batch_size:(i + 1) * self.batch_size]
					z_batch = z_data[i * self.batch_size:(i + 1) * self.batch_size]
					feed_dict = {x_ph: x_batch, z_ph: z_batch}
					if epoch + i == 0:
						_ = sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
					_ = sess.run(train_ops['critic'], feed_dict=feed_dict) 
					info = self.get_info(sess, feed_dict, quantities, epoch, i)
					if ground_truth_mi is not None:
						info.append("true_mi={:.3g}".format(ground_truth_mi))
					batch_bar.set_description('   '.join(info))
					mi_batch = sess.run(quantities['mi'], feed_dict=feed_dict)
					mi_ema -= (1 - 0.9) * (mi_ema - mi_batch)
		return mi_ema

