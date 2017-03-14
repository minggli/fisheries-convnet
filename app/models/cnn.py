import tensorflow as tf

class ConvolutionalNeuralNet(object):

    def __init__(self, shape):
        """shape: [n_samples, channels, n_features]"""
        self.shape = shape

    @staticmethod
    def weight_variable(shape):
        initial = tf.random_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def non_linearity(activation_func):
        if activation_func == 'sigmoid':
            return tf.nn.sigmoid
        elif activation_func == 'relu':
            return tf.nn.relu

    @property
    def x(self):
        """feature set"""
        return tf.placeholder(
            dtype=tf.float32, shape=[None, self.shape[1], self.shape[2]],
            name='feature'
        )

    @property
    def _y(self):
        """true label, in one hot format"""
        return tf.placeholder(
            dtype=tf.float32, shape=[None, self.shape[2]], name='label'
        )

    def add_conv_layer(self, x, hyperparams, func='relu'):
        """Convolution Layer with hyperparamters and activation_func"""
        W = self.__class__.weight_variable(shape=hyperparams[0])
        b = self.__class__.bias_variable(shape=hyperparams[1])

        hypothesis_conv = self.__class__.non_linearity(func)(
            self.__class__.conv2d(x, W) + b)
        hypothesis_pool = self.__class__.max_pool(hypothesis_conv)
        return hypothesis_pool

    def add_dense_layer(self, x, hyperparams, func='relu'):
        """Densely Connected Layer with hyperparamters and activation_func"""
        W = self.__class__.weight_variable(shape=hyperparams[0])
        b = self.__class__.bias_variable(shape=hyperparams[1])

        flat_x = tf.reshape(x, hyperparams[2])
        hypothesis = \
            self.__class__.non_linearity(func)(tf.matmul(flat_x, W) + b)
        return hypothesis

    def add_drop_out_layer(self, x):
        """drop out layer to reduce overfitting"""
        keep_prob = tf.placeholder(dtype=tf.float32)
        hypothesis_drop = tf.nn.dropout(x, keep_prob)
        return hypothesis_drop

    def add_read_out_layer(self, x, hyperparams):
        """final read out layer"""
        W = self.__class__.weight_variable(shape=hyperparams[0])
        b = self.__class__.bias_variable(shape=hyperparams[1])
        
        logits = tf.matmul(x, W) + b
        return logits
