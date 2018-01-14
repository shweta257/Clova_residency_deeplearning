import tensorflow as tf
import os, sys
import numpy as np
from config import FLAGS
from tqdm import tqdm
from tensorflow.contrib.layers import conv2d, fully_connected

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

epsilon = 1e-9

def load_mnist(path):
	fd = open(os.path.join(FLAGS.dataset, 'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

	fd = open(os.path.join(FLAGS.dataset, 'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	trainY = loaded[8:].reshape((60000)).astype(np.int32)

	fd = open(os.path.join(FLAGS.dataset, 't10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	testX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

	fd = open(os.path.join(FLAGS.dataset, 't10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	testY = loaded[8:].reshape((10000)).astype(np.int32)

	trainX = tf.convert_to_tensor(trainX / 255., tf.float32)

	return trainX, trainY, testX / 255., testY

#create batches
def create_batch(x, y):
	data = tf.train.slice_input_producer([x, y])
	X, Y = tf.train.shuffle_batch(data, num_threads=4,
								  batch_size=FLAGS.batch_size,
								  capacity=FLAGS.batch_size * 64,
								  min_after_dequeue=FLAGS.batch_size * 32,
								  allow_smaller_final_batch=False)

	return (X, Y)

#define model
def model_capsule(X, Y):
	with tf.variable_scope('Conv_layer'):
		conv1 = conv2d(X, num_outputs=256,kernel_size=9, stride=1,padding='VALID')
	
	with tf.variable_scope('PrimaryCaps_layer'):
		prime_caps = conv2d(conv1, num_outputs=256, kernel_size=9, stride=2, padding="VALID")
		prime_caps = tf.reshape(prime_caps, (FLAGS.batch_size, -1, 8, 1))
		prime_caps = squash(prime_caps)

  	with tf.variable_scope('DigitCaps_layer'):
  		prime_caps = tf.reshape(prime_caps, shape=(FLAGS.batch_size, -1, 1, prime_caps.shape[-2].value, 1))
  		with tf.variable_scope('routing'):
  			b_ij = tf.constant(np.zeros([1, prime_caps.shape[1].value, 10, 1, 1], dtype=np.float32))
			#routing
			digit_caps = routing(prime_caps, b_ij)
			digit_caps = tf.squeeze(digit_caps, axis=1)
	
	with tf.variable_scope('Masking'):
		#masking
		masked_v = tf.matmul(tf.squeeze(digit_caps), tf.reshape(Y, (-1, 10, 1)), transpose_a=True)
		predict = tf.sqrt(tf.reduce_sum(tf.square(digit_caps), axis=2, keep_dims=True) + epsilon)

	with tf.variable_scope('Decoder'):
		v_j = tf.reshape(masked_v, shape=(FLAGS.batch_size, -1))
		fc_layer1 = fully_connected(v_j, num_outputs=512)
		fc_layer2 = fully_connected(fc_layer1, num_outputs=1024)
		reconstructed = fully_connected(fc_layer2, num_outputs=784, activation_fn=tf.sigmoid)

	return predict, reconstructed

#define routing
def routing(input, b_ij):
	W = tf.get_variable('Weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=FLAGS.stddev))
	input = tf.tile(input, [1, 1, 10, 1, 1])
	W = tf.tile(W, [FLAGS.batch_size, 1, 1, 1, 1])
	u_hat = tf.matmul(W, input, transpose_a=True)
	for r_iter in range(FLAGS.route_iter):
		with tf.variable_scope('iter_' + str(r_iter)):
			c_ij = tf.nn.softmax(b_ij, dim=2)
			c_ij = tf.tile(c_ij, [FLAGS.batch_size, 1, 1, 1, 1])
			s_j = tf.multiply(c_ij, u_hat)
			s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)
			v_j = squash(s_j)
			v_j_tiled = tf.tile(v_j, [1, 1152, 1, 1, 1])
			u_dot_v = tf.matmul(u_hat, v_j_tiled, transpose_a=True)
			b_ij += tf.reduce_sum(u_dot_v, axis=0, keep_dims=True)
	return (v_j)

#define squashing
def squash(v):
	v_squared_norm = tf.reduce_sum(tf.square(v), -2, keep_dims=True)
	scalar_factor = v_squared_norm / (1 + v_squared_norm) / tf.sqrt(v_squared_norm + epsilon)
	v_squashed = scalar_factor * v 
	return (v_squashed)

def loss(X, Y, predict, reconstructed):
	max_l = tf.square(tf.maximum(0., FLAGS.m_plus - predict))
	max_r = tf.square(tf.maximum(0., predict - FLAGS.m_minus))
	max_l = tf.reshape(max_l, shape=(FLAGS.batch_size, -1))
	max_r = tf.reshape(max_r, shape=(FLAGS.batch_size, -1))
	
	T_c = Y
	L_c = T_c * max_l + FLAGS.lambda_val * (1 - T_c) * max_r
	margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
	
	orgin = tf.reshape(X, shape=(FLAGS.batch_size, -1))
	squared = tf.square(reconstructed - orgin)
	recon_err = tf.reduce_mean(squared)	

	total_loss = margin_loss + FLAGS.reg_scale * recon_err
	return margin_loss, recon_err, total_loss

def calc_accuracy(label, predict):
	softmax_v = tf.nn.softmax(predict, dim=1)
	argmax_index = tf.to_int32(tf.argmax(softmax_v, axis=1))
	argmax_index = tf.reshape(argmax_index, shape=(FLAGS.batch_size, ))

	right_predict = tf.equal(tf.to_int32(label), tf.to_int32(predict))
	acc = tf.reduce_sum(tf.cast(right_predict, tf.float32))
	return acc

def main(_):
	#load dataset
	trainX, trainY, testX, testY = load_mnist(FLAGS.dataset)

	X, label = create_batch(trainX, trainY)
	Y = tf.one_hot(label, depth=10, axis=1, dtype=tf.float32)

	predict, reconstructed = model_capsule(X, Y)

	#define loss
	margin_loss, reconstruction_loss, total_loss = loss(X, Y, predict, reconstructed)

	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer().minimize(total_loss) 

	batch_accuracy = calc_accuracy(trainY, predict)

	#summary
	tf.summary.scalar('margin_loss', margin_loss)
	tf.summary.scalar('reconstruction_loss', reconstruction_loss)
	tf.summary.scalar('total_loss', total_loss)
	
	recon_img = tf.reshape(reconstructed, shape=(FLAGS.batch_size, 28, 28, 1))
	tf.summary.image('reconstruction_img', recon_img)
	tf.summary.scalar("training_accuracy", batch_accuracy)
	merged = tf.summary.merge_all()

	#do training
	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
		test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

		num_batch = int(60000 / FLAGS.batch_size)
		num_test_batch = int(10000 / FLAGS.batch_size)
		
		for epoch in range(FLAGS.epoch):
			for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
				sess.run(train_step)
				if step % FLAGS.train_sum_freq == 0:
					summary, _ = sess.run([merged, train_step])
					
					train_writer.add_summary(summary, step)
				if (step + 1) % FLAGS.test_sum_freq == 0:
					test_acc = 0
					total_test_acc = 0

					for i in range(num_test_batch):
						start = i * FLAGS.batch_size
						end = start + FLAGS.batch_size
						summary, test_acc = sess.run([merged, accuracy], {X: testX[start:end], Y: testY[start:end]})
						total_test_acc += test_acc
						test_writer.add_summary(summary, i)

					test_acc = total_test_acc / (FLAGS.batch_size * num_test_batch)
					print(str(step + 1) + ',' + str(test_acc) + '\n')
			if epoch % FLAGS.save_freq == 0:
				saver.save(sess, '.' + '/model_epoch_%04d_step_%02d' % (epoch, step))

	#use summary for train and test loss


if __name__ == "__main__":
	tf.app.run()
