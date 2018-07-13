import tensorflow as tf
import numpy as np
import scipy.io as io

PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'max'
IMAGE = 'me.jpg'



def get_convfilter(layers_data, i):
    data = layers_data[i][0][0][0][0][0]
    d = tf.constant(data)
    return d


def get_biases(layers_data, i):
    data = layers_data[i][0][0][0][0][1]
    d = tf.constant(data)
    return d


def conv_layer(layers_data, i, bottom):
    filt = get_convfilter(layers_data, i)
    return tf.nn.conv2d(bottom, filt, strides=[1, 1, 1, 1], padding='SAME')


def pooling_kind(pooling, bottom):
    if pooling == 'max':
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    elif pooling == 'avg':
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def relu_layer(layer_input, bias):
    return tf.nn.relu(layer_input + bias)

def load_network(path):
    vgg_data = io.loadmat(path)
    if not all(i in vgg_data for i in ('layers', 'classes', 'normalization')):
        raise ValueError('Wrong VGG data')
    layers_weights = vgg_data['layers'][0]
    return layers_weights


def load_weights(layers_weights, img, pooling):

    net = {'conv1_1': conv_layer(layers_weights, 0, img)}

    net['relu1_1'] = relu_layer(net['conv1_1'], bias=get_biases(layers_weights, 0))

    net['conv1_2'] = conv_layer(layers_weights, 2, net['relu1_1'])
    net['relu1_2'] = relu_layer(net['conv1_2'], bias=get_biases(layers_weights, 2))

    net['pool1'] = pooling_kind(pooling, net['relu1_2'])

    net['conv2_1'] = conv_layer(layers_weights, 5, net['pool1'])
    net['relu2_1'] = relu_layer(net['conv2_1'], bias=get_biases(layers_weights, 5))

    net['conv2_2'] = conv_layer(layers_weights, 7, net['relu2_1'])
    net['relu2_2'] = relu_layer(net['conv2_2'], bias=get_biases(layers_weights, 7))

    net['pool2'] = pooling_kind(pooling, net['relu2_2'])

    net['conv3_1'] = conv_layer(layers_weights, 10, net['pool2'])
    net['relu3_1'] = relu_layer(net['conv3_1'], bias=get_biases(layers_weights, 10))

    net['conv3_2'] = conv_layer(layers_weights, 12, net['relu3_1'])
    net['relu3_2'] = relu_layer(net['conv3_2'], bias=get_biases(layers_weights, 12))

    net['conv3_3'] = conv_layer(layers_weights, 14, net['relu3_2'])
    net['relu3_3'] = relu_layer(net['conv3_3'], bias=get_biases(layers_weights, 14))

    net['conv3_4'] = conv_layer(layers_weights, 16, net['relu3_3'])
    net['relu3_4'] = relu_layer(net['conv3_4'], bias=get_biases(layers_weights, 16))

    net['pool3'] = pooling_kind(pooling, net['relu3_4'])

    net['conv4_1'] = conv_layer(layers_weights, 19, net['pool3'])
    net['relu4_1'] = relu_layer(net['conv4_1'], bias=get_biases(layers_weights, 19))

    net['conv4_2'] = conv_layer(layers_weights, 21, net['relu4_1'])
    net['relu4_2'] = relu_layer(net['conv4_2'], bias=get_biases(layers_weights, 21))

    net['conv4_3'] = conv_layer(layers_weights, 23, net['relu4_2'])
    net['relu4_3'] = relu_layer(net['conv4_3'], bias=get_biases(layers_weights, 23))

    net['conv4_4'] = conv_layer(layers_weights, 25, net['relu4_3'])
    net['relu4_4'] = relu_layer(net['conv4_4'], bias=get_biases(layers_weights, 25))

    net['pool4'] = pooling_kind(pooling, net['relu4_4'])

    net['conv5_1'] = conv_layer(layers_weights, 28, net['pool4'])
    net['relu5_1'] = relu_layer(net['conv5_1'], bias=get_biases(layers_weights, 28))

    net['conv5_2'] = conv_layer(layers_weights, 30, net['relu5_1'])
    net['relu5_2'] = relu_layer(net['conv5_2'], bias=get_biases(layers_weights, 30))

    net['conv5_3'] = conv_layer(layers_weights, 32, net['relu5_2'])
    net['relu5_3'] = relu_layer(net['conv5_3'], bias=get_biases(layers_weights, 32))

    net['conv5_4'] = conv_layer(layers_weights, 34, net['relu5_3'])
    net['relu5_4'] = relu_layer(net['conv5_4'], bias=get_biases(layers_weights, 34))

    net['pool5'] = pooling_kind(pooling, net['relu5_4'])

    return net

