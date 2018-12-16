import tensorflow as tf
import numpy as np
from ops import  *
import os
from glob import glob
import time


class Hidden(object):
	def __init__(self, sess, input_height = 16, input_width = 16, batch_size = 64, output_height = 16, output_width = 16, z_len = 52, img_dim = 3, dataset_name = 'celeba', epoch = 40):
		self.sess = sess
		self.lambda1 = self.lambda2 = 0.2
		self.input_height = input_height
		self.input_width = input_width
		self.output_height = output_height
		self.output_width = output_width
		self.batch_size = batch_size
		self.z_len = z_len
		self.img_dim = img_dim
		self.dataset_name = dataset_name
		self.epoch = epoch

		self.data = glob(os.path.join('./data', dataset_name, "*.jpg"))

		self.de_bn1 = batch_norm(name='de_bn1')
		self.de_bn2 = batch_norm(name='de_bn2')
		self.de_bn3 = batch_norm(name='de_bn3')
		self.de_bn4 = batch_norm(name='de_bn4')
		self.de_bn5 = batch_norm(name='de_bn5')
		self.de_bn6 = batch_norm(name='de_bn6')
		self.de_bn7 = batch_norm(name='de_bn7')

		self.en_bn1 = batch_norm(name='en_bn1')
		self.en_bn2 = batch_norm(name='en_bn2')
		self.en_bn3 = batch_norm(name='en_bn3')
		self.en_bn4 = batch_norm(name='en_bn4')
		self.en_bn5 = batch_norm(name='en_bn5')
		self.en_bn6 = batch_norm(name='en_bn6')
		self.en_bn7 = batch_norm(name='en_bn7')

		self.dis_bn1 = batch_norm(name='dis_bn1')
		self.dis_bn2 = batch_norm(name='dis_bn2')
		self.dis_bn3 = batch_norm(name='dis_bn3')
		self.dis_bn4 = batch_norm(name='dis_bn4')
		self.dis_bn5 = batch_norm(name='dis_bn5')
		self.dis_bn6 = batch_norm(name='dis_bn6')
		self.dis_bn7 = batch_norm(name='dis_bn7')
		self.build_model()

	def build_model(self):
		self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, self.img_dim], name = "input_img")
		self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_len], name = "z")
		self.mutil_z = tf.placeholder(tf.float32,[self.batch_size, self.input_height, self.input_width, self.z_len])
		self.E = self.encoder(self.mutil_z)
		self.D = self.decoder()

		self.ori_A = self.distriminator(self.inputs)
		self.fak_A = self.distriminator(self.E)

		self.E_sum = histogram_summary("e_", self.E)
		self.D_sum = histogram_summary("d_", self.D)
		self.ori_A_sum = histogram_summary("ori_", self.ori_A)
		self.fak_A_sum = histogram_summary("fak_", self.fak_A)


		self.en_loss = tf.losses.mean_squared_error(self.E, self.inputs)/(self.input_height*self.input_width*self.img_dim)
		self.de_loss = tf.losses.mean_squared_error(self.z, self.D)
		self.dis_loss = tf.reduce_mean(distriminator_loss(self. fak_A))
		self.cls_loss = tf.reduce_mean(distriminator_loss(self.ori_A) + tf.log(self.fak_A))

		self.loss_1 = self.de_loss+ self.lambda1*self.en_loss+self.lambda2*self.dis_loss
		self.loss_2 = self.cls_loss

		self.acc = tf.reduce_mean(tf.cast(tf.equal(self.z, tf.round(self.D)), tf.float32))

		self.acc_sum = scalar_summary("acc", self.acc)

		self.en_loss_sum = scalar_summary("en_loss", self.en_loss)
		self.de_loss_sum = scalar_summary("de_loss", self.de_loss)
		self.dis_loss_sum = scalar_summary("dis_loss", self.dis_loss)
		self.cls_loss_sum = scalar_summary("cls_loss", self.cls_loss)
		self.loss_1_sum = scalar_summary("loss_1", self.loss_1)
		self.loss_2_sum = scalar_summary("loss_2", self.loss_2)

		self.all_vars = tf.trainable_variables()

		self.de_vars = [var for var in self.all_vars if "de_" in var.name]
		self.en_vars = [var for var in self.all_vars if "en_" in var.name]
		self.dis_vars = [var for var in self.all_vars if "dis_" in var.name]

		self.z_sum = histogram_summary("z", self.z)

		self.saver = tf.train.Saver()

	def train(self):
		optim_1 = tf.train.AdamOptimizer().minimize(self.loss_1, var_list = self.all_vars)
		optim_2 = tf.train.AdamOptimizer().minimize(self.loss_2, var_list = self.dis_vars)

		try:
			tf.global_variables_initializer().run()
		except:
			tf.initialize_all_variables().run()

		self.en_sum = merge_summary([self.z_sum, self.E_sum, self.en_loss_sum, self.dis_loss_sum, self.fak_A_sum, self.loss_1_sum, self.loss_2_sum])
		self.de_sum = merge_summary([self.z_sum, self.D_sum, self.de_loss_sum, self.acc_sum, self.loss_1_sum])
		self.dis_sum = merge_summary([self.ori_A_sum, self.z_sum, self.fak_A_sum, self.dis_loss_sum, self.cls_loss_sum, self.loss_1_sum, self.loss_2_sum])

		self.writer = SummaryWriter("./logs", self.sess.graph)
		sample_z = np.random.uniform(0, 1, [self.batch_size, self.z_len]).astype(np.float32)
		sample_z = np.round(sample_z).astype(np.float32)

		counter = 1
		start_time = time.time()
		for epoch in xrange(self.epoch):
			batch_idx = len(self.data) // self.batch_size
			for i in xrange(0, batch_idx):
				batch_files = self.data[i*self.batch_size:(i+1)*self.batch_size]
				batch = [get_image(batch_file,
								   input_height = self.input_height,
								   input_width = self.input_width,
								   resize_height = self.input_height,
								   resize_width = self.output_width,
								   crop = True,
								   graysacle=False
								   ) for batch_file in batch_files]
				batch_img = np.array(batch).astype(np.float32)
				batch_z = np.random.uniform(0, 1, [self.batch_size, self.z_len]).astype(np.float32)
				batch_z = np.round(batch_z)
				tmp_2d = np.zeros(shape=[self.input_height, self.z_len])
				tmp_3d = np.zeros(shape=[self.input_height, self.z_len, self.input_width])
				batch_mutil_z = np.zeros(shape=[self.batch_size, self.input_height, self.input_width, self.z_len])
				idx = 0

				for z in batch_z:
					for i in range(self.input_height):
						tmp_2d[i, :] = z
					for i in range(self.input_width):
						tmp_3d[:, :, i] = tmp_2d
					mutil_z = tf.reshape(tmp_3d, shape=[self.input_height, self.input_width, self.z_len])
					batch_mutil_z[idx,:,:,:] = mutil_z
					idx+=1
				batch_mutil_z = batch_mutil_z.astype(np.float32)
				_, summary_str = self.sess.run([optim_1, self.en_sum, self.de_sum, self.dis_sum],
											   feed_dict={self.inputs: batch_img, self.z: batch_z, self.mutil_z: batch_mutil_z})
				self.writer.add_summary(summary_str, counter)
				_, summary_str = self.sess.run([optim_2, self.en_sum, self.dis_sum],
											   feed_dict={self.inputs: batch_img, self.z: batch_z, self.mutil_z: batch_mutil_z})
				self.writer.add_summary(summary_str, counter)
				loss_1, loss_2, acc = self.sess.run([self.loss_1, self.loss_2, self.acc], feed_dict={self.inputs:batch_img, self.z:batch_z, self.mutil_z: batch_mutil_z} )
				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss_1: %.8f, loss_2: %.8f, acc: %.8f " \
				  % (epoch, i, batch_idx, time.time() - start_time, loss_1, loss_2, acc))

	def encoder(self, mutil_z):
		with tf.variable_scope("en_model") as scope:
			layer_1 = tf.nn.relu(self.en_bn1(conv2d(self.inputs, 64, name="en_conv_1")))
			layer_2 = tf.nn.relu(self.en_bn2(conv2d(layer_1, 64, name="en_conv_2")))
			layer_3 = tf.nn.relu(self.en_bn3(conv2d(layer_2, 64, name="en_conv_3")))
			layer_4 = tf.nn.relu(self.en_bn4(conv2d(layer_3, 64, name="en_conv_4")))
			print(layer_4.dtype, self.inputs.dtype, mutil_z.dtype)
			united_layer = tf.concat([self.inputs, layer_4, mutil_z],3)
			layer_6 = tf.nn.relu(self.en_bn6(conv2d(united_layer, 64, name="en_conv_6")))
			fake_img = conv2d(layer_6, 3, k_h=1, k_w=1, d_h=1, d_w=1)

		return fake_img

	def decoder(self):
		with tf.variable_scope("de_model") as scope:
			layer_1 = tf.nn.relu(self.de_bn1(conv2d(self.E, 64, name="de_conv_1")))
			layer_2 = tf.nn.relu(self.de_bn2(conv2d(layer_1, 64, name="de_conv_2")))
			layer_3 = tf.nn.relu(self.de_bn3(conv2d(layer_2, 64, name="de_conv_3")))
			layer_4 = tf.nn.relu(self.de_bn4(conv2d(layer_3, 64, name="de_conv_4")))
			layer_5 = tf.nn.relu(self.de_bn5(conv2d(layer_4, 64, name="de_conv_5")))
			layer_6 = tf.nn.relu(self.de_bn6(conv2d(layer_5, 64, name="de_conv_6")))
			layer_7 = tf.nn.relu(self.de_bn7(conv2d(layer_6, self.z_len, name="de_conv_7")))
			avg_layer = tf.layers.average_pooling2d(inputs=layer_7, pool_size=[16,16], strides=1, padding='VALID')
			avg_layer = tf.reshape(avg_layer,[self.batch_size, self.z_len])
			print("avgsize", avg_layer.shape)
			res = linear(avg_layer, self.z_len, "de_ln")

		return res

	def distriminator(self, img):
		with tf.variable_scope("dis_model", reuse=tf.AUTO_REUSE) as scope:
			layer_1 = tf.nn.relu(self.dis_bn1(conv2d(img, 64, name="dis_conv_1")))
			layer_2 = tf.nn.relu(self.dis_bn2(conv2d(layer_1, 64, name="dis_conv_2")))
			layer_3 = tf.nn.relu(self.dis_bn3(conv2d(layer_2, 64, name="dis_conv_3")))
			avg_layer = tf.layers.average_pooling2d(inputs=layer_3, pool_size=[16,16], strides=1, padding='VALID')
			avg_layer = tf.reshape(avg_layer,[self.batch_size, 64])
			print("avgsize", avg_layer.shape)
			res = linear(avg_layer, 2, "dis_ln")
		return res

def distriminator_loss(fa):
	return tf.log(1 - fa)
