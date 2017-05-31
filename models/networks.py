import tensorflow as tf


def relu(input_):
    with tf.name_scope('relu'):
        return tf.nn.relu(input_)


def lrelu(input_, leak=0.2):
    with tf.name_scope('lrelu'):
        return tf.maximum(input_, leak * input_)


def instance_norm(input_):
    with tf.name_scope('instance_norm'):
        num_channels = input_.get_shape()[-1]
        scale = tf.get_variable(name="scale",
                                shape=[num_channels],
                                initializer=tf.random_normal_initializer(mean=1.0, stddev=2e-2, dtype=tf.float32))

        offset = tf.get_variable(name='offset',
                                 shape=[num_channels],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=2e-2, dtype=tf.float32))

        epsilon = 1e-9
        mean, var = tf.nn.moments(input_, [1, 2], keep_dims=True)
        normalized = tf.div(tf.subtract(input_, mean), tf.sqrt(tf.add(var, epsilon)))

        return scale * normalized + offset


def batch_norm(input_, is_training):
    with tf.name_scope('batch_norm'):
        return tf.contrib.layers.batch_norm(input_, scale=True, updates_collections=None, is_training=is_training)


def conv(input_, out_channels, kernel_size, stride, stddev=2e-2):
    with tf.variable_scope('conv'):
        in_channels = input_.get_shape()[-1]
        filter_ = tf.get_variable(
            name='filter',
            shape=[kernel_size, kernel_size, in_channels, out_channels],
            initializer=tf.truncated_normal_initializer(stddev=stddev),
        )
        conv = tf.nn.conv2d(input_, filter_, [1, stride, stride, 1], padding='SAME')
        return conv


def deconv(input_, out_channels, kernel_size, stride, stddev=2e-2):
    with tf.variable_scope("deconv"):
        _, in_height, in_width, in_channels = input_.get_shape().as_list()
        filter_ = tf.get_variable(
            name='filter',
            shape=[kernel_size, kernel_size, out_channels, in_channels],
            initializer=tf.truncated_normal_initializer(stddev=stddev),
        )

        batch_dynamic = tf.shape(input_)[0]
        output_shape = [batch_dynamic, in_height * 2, in_width * 2, out_channels]
        conv = tf.nn.conv2d_transpose(input_, filter_, tf.stack(output_shape), [1, stride, stride, 1], padding="SAME")
        conv = tf.reshape(conv, [-1, output_shape[1], output_shape[2], output_shape[3]])
        return conv


class ConvBlock:
    def __init__(self, name, kernel_size, stride, out_channels, activation_fn, use_norm=True):
        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels
        self.activation_fn = activation_fn
        self.use_norm = use_norm

    def forward(self, input_):
        with tf.variable_scope(self.name):
            output = conv(input_, self.out_channels, self.kernel_size, self.stride)
            if self.use_norm:
                output = instance_norm(output)
            if self.activation_fn is not None:
                output = self.activation_fn(output)
            return output


class ResidualBlock:
    def __init__(self, name, kernel_size, stride, out_channels):
        self.name = name
        self.conv_blocks = [
            ConvBlock('conv0', kernel_size, stride, out_channels, relu, True),
            ConvBlock('conv1', kernel_size, stride, out_channels, None, True)
        ]

    def forward(self, input_):
        with tf.variable_scope(self.name):
            conv_output = None
            for block in self.conv_blocks:
                conv_output = block.forward(input_)

            output = input_ + conv_output
            return output


class DeconvBlock:
    def __init__(self, name, kernel_size, stride, out_channels):
        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels

    def forward(self, input_):
        with tf.variable_scope(self.name):
            output = deconv(input_, self.out_channels, self.kernel_size, self.stride)
            output = instance_norm(output)
            output = tf.nn.relu(output)
            return output


class Generator:
    def __init__(self, name, out_channels):
        self.name = name
        self.used = False
        self.model = [
            ConvBlock('block_00_c7s1-32', 7, 1, 32, relu),
            ConvBlock('block_01_d64', 3, 2, 64, relu),
            ConvBlock('block_02_d128', 3, 2, 128, relu),
            ResidualBlock('block_03_R128', 3, 1, 128),
            ResidualBlock('block_04_R128', 3, 1, 128),
            ResidualBlock('block_05_R128', 3, 1, 128),
            ResidualBlock('block_06_R128', 3, 1, 128),
            ResidualBlock('block_07_R128', 3, 1, 128),
            ResidualBlock('block_08_R128', 3, 1, 128),
            ResidualBlock('block_09_R128', 3, 1, 128),
            ResidualBlock('block_10_R128', 3, 1, 128),
            ResidualBlock('block_11_R128', 3, 1, 128),
            DeconvBlock('block_12_u64', 3, 2, 64),
            DeconvBlock('block_13_u32', 3, 2, 32),
            ConvBlock('block_14_c7s1-3', 7, 1, out_channels, relu),
        ]

    def forward(self, input_):
        with tf.variable_scope(self.name) as scope:
            if self.used:
                scope.reuse_variables()
            else:
                self.used = True

            output = input_
            for block in self.model:
                output = block.forward(output)

            return tf.nn.tanh(output)


class Discriminator:
    def __init__(self, name):
        self.name = name
        self.used = False
        self.model = [
            ConvBlock('block_0_C64', 4, 2, 64, lrelu, use_norm=False),
            ConvBlock('block_1_C128', 4, 2, 128, lrelu),
            ConvBlock('block_2_C256', 4, 2, 256, lrelu),
            ConvBlock('block_3_C512', 4, 2, 512, lrelu),
            ConvBlock('block_4', 4, 1, 1, None, use_norm=False),
        ]

    def forward(self, input_):
        with tf.variable_scope(self.name) as scope:
            if self.used:
                scope.reuse_variables()
            else:
                self.used = True

            output = input_
            for block in self.model:
                output = block.forward(output)

            # return tf.nn.sigmoid(output)
            return output
