import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('dataset', 'data/mnist', 'the path for dataset')
flags.DEFINE_string('summaries_dir', 'summaries_dir', 'summary directory')
flags.DEFINE_integer('train_sum_freq', 50, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('test_sum_freq', 500, 'the frequency of saving test summary(step)')
flags.DEFINE_integer('save_freq', 5, 'the frequency of saving model(epoch)')
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epoch', 20, 'number of epoch')
flags.DEFINE_integer('route_iter', 3, 'number of iterations for routing')
flags.DEFINE_float('stddev', 0.01, 'standard deviation for W initialization')
flags.DEFINE_float('reg_scale', 0.0005, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')


FLAGS = tf.app.flags.FLAGS