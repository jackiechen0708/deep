import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
import numpy as np

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)

WEIGHT_DECAY = 0.0005
DROPOUT = 0.5
NUM_CLASSES = 10
NUM_OF_EPOCH = 80
BATCH_SIZE = 128
INIT_LR = 0.01
MOMENTUM = 0.9
STD = 64
ERR = 1e-6


def normalize(train, test):
    mean = np.mean(train, axis=(0, 1, 2, 3))
    std = np.std(train, axis=(0, 1, 2, 3))
    return (train - mean) / (std + ERR), (test - mean) / (std + ERR)

def lenet(input,
          is_training=True,
          dropout_keep_prob=0.5,
          ):
    with tf.variable_scope("lenet", reuse=tf.AUTO_REUSE):
        with arg_scope(
                [layers.conv2d, layers_lib.max_pool2d],
                data_format="NCHW"):
            x = layers.conv2d(
                input,
                32,
                kernel_size=[5, 5],
                padding="same",
                activation_fn=tf.nn.relu)
            x = layers_lib.max_pool2d(x, [2, 2], 2)
            x = layers.conv2d(
                x,
                64,
                kernel_size=[5, 5],
                padding="same",
                activation_fn=tf.nn.relu)
            x = layers_lib.max_pool2d(x, [2, 2], 2)
            print (x)
            x = tf.reshape(x, [-1, 8 * 8 * 64])
            x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
            x = tf.layers.dropout(
                inputs=x, rate=dropout_keep_prob, training=is_training)
            x = tf.layers.dense(inputs=x, units=10)
        return x

def alexnet(inputs,
            is_training=True,
            dropout_keep_prob=0.5, ):
    with tf.variable_scope("alexnet", reuse=tf.AUTO_REUSE):
        with arg_scope(
                [layers.conv2d, layers_lib.max_pool2d],
                data_format="NCHW"):
            x = layers.conv2d(
                inputs, 64, [3, 3],  padding='VALID', scope='conv1')
            x = layers_lib.max_pool2d(x, [2, 2], 2, scope='pool1')
            x = layers.conv2d(x, 192, [5, 5], scope='conv2')
            x = layers_lib.max_pool2d(x, [2, 2], 2, scope='pool2')
            x = layers.conv2d(x, 384, [3, 3], scope='conv3')
            x = layers.conv2d(x, 384, [3, 3], scope='conv4')
            x = layers.conv2d(x, 256, [3, 3], scope='conv5')
            x = layers_lib.max_pool2d(x, [2, 2], 2, scope='pool5')

            with arg_scope(
                    [layers.conv2d],
                    weights_initializer=trunc_normal(0.005),
                    biases_initializer=init_ops.constant_initializer(0.1)):
                x = layers.conv2d(x, 4096, [3, 3], padding='VALID', scope='fc6')
                x = layers_lib.dropout(x, dropout_keep_prob, is_training=is_training, scope='dropout6')
                x = layers.conv2d(x, 4096, [1, 1], scope='fc7')
                x = layers_lib.dropout(x, dropout_keep_prob, is_training=is_training, scope='dropout7')
                x = layers.conv2d(x, NUM_CLASSES, [1, 1], activation_fn=None, normalizer_fn=None,
                                  biases_initializer=init_ops.zeros_initializer(), scope='fc8')
                x = tf.squeeze(x, [2, 3], name='fc8/squeezed')

        return x


image_placeholder = tf.placeholder(tf.float32, shape=[None, 3, 32, 32])
label_placeholder = tf.placeholder(tf.float32, shape=[None, 10])
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train, x_test = normalize(x_train, x_test)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


x_train,y_train = tf.train.shuffle_batch(
      [x_train,y_train],
      batch_size=BATCH_SIZE,
      num_threads=1,capacity=2000,min_after_dequeue=1,enqueue_many=True,allow_smaller_final_batch=True)


with arg_scope(
        [layers.conv2d],
        activation_fn=nn_ops.relu,
        biases_initializer=init_ops.constant_initializer(0.1),
        weights_regularizer=regularizers.l2_regularizer(WEIGHT_DECAY)):
    with arg_scope([layers.conv2d], padding='SAME'):
        with arg_scope([layers_lib.max_pool2d], padding='VALID') as arg_sc:
            #output = alexnet(x_train)
            output = lenet(x_train)
            #output_test = alexnet(image_placeholder, is_training=False)
            output_test = lenet(image_placeholder, is_training=False)
global_step = tf.Variable(0, trainable=False)
loss = tf.losses.softmax_cross_entropy(y_train, output)
lr = tf.train.exponential_decay(INIT_LR, global_step, 1000, 0.96)
opt = tf.train.MomentumOptimizer(lr, MOMENTUM)
train_op = opt.minimize(loss, global_step)

saver = tf.train.Saver()
init = tf.initialize_all_variables()
def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        #pylint: disable=W0212
        session = session._sess
    return session
with tf.train.SingularMonitoredSession() as sess:
    sess.run(init)
    for i in range(NUM_OF_EPOCH):
        for j in range(50000 // BATCH_SIZE):
            loss_val,lr_val,output_val,_=sess.run([loss,lr,output,train_op] )
            print ("EPOCH:"+str(i),"ITER:"+str(j),loss_val)
            #print (lr_val)
            #print (output_val)
        predicted_x = sess.run(output_test, feed_dict={image_placeholder: x_test})
        residuals = np.argmax(predicted_x, 1) == np.argmax(y_test, 1)
        accr = 1.0 * sum(residuals) / len(residuals)
        print ("EPOCH:"+str(i),"Top1 Accuracy:"+str(accr))

        saver.save(get_session(sess), "./lenet-model")
