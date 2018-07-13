

import vgg19

import tensorflow as tf
import numpy as np
import image_handler as ih
from sys import stderr

from PIL import Image
from functools import reduce

CONTENT_LAYERS = ('conv4_2', 'conv5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


def layer_weight_normalize(weights, layer_weight, weight_exp):
    layer_weights_sum = 0
    for layer in STYLE_LAYERS:
        weights[layer] = layer_weight
        layer_weight *= weight_exp
        layer_weights_sum += weights[layer]
    for layer in STYLE_LAYERS:
        weights[layer] /= layer_weights_sum

    return weights


def stylize(network, initial, initial_noiseblend, content, styles, preserve_colors, iterations, content_weight,
            content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
            learning_rate, beta1, beta2, epsilon, pooling, print_iterations=None, checkpoint_iterations=None):

    vgg_weights = vgg19.load_network(network)
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]
    layer_weight = 1.0
    style_layers_weights = {}
    style_layers_weights = layer_weight_normalize(style_layers_weights, layer_weight, style_layer_weight_exp)

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/gpu:0'), tf.Session() as s:
        image = tf.placeholder('float', shape=shape)
        net = vgg19.load_weights(vgg_weights, image, pooling)
        content_pre = np.array([ih.preprocess(content)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre}, session=s)

    # compute style features in feedforward mode
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/gpu:0'), tf.Session() as s:
            image = tf.placeholder('float', shape=style_shapes[i])
            net = vgg19.load_weights(vgg_weights, image, pooling)
            style_pre = np.array([ih.preprocess(styles[i])])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre}, session=s)
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                style_features[i][layer] = gram

    initial_content_noise_coeff = 1.0 - initial_noiseblend

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        if initial is None:
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([ih.preprocess(initial)])
            initial = initial.astype('float32')
            initial = initial * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (1.0 - initial_content_noise_coeff)
        image = tf.Variable(initial)
        net = vgg19.load_weights(vgg_weights, image, pooling)

        # content loss
        content_layers_weights = {'conv4_2': content_weight_blend, 'conv5_2': 1.0 - content_weight_blend}

        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            _, height, width, number = map(lambda j: j.value, net[content_layer].get_shape())
            gram_diff = tf.reduce_sum(tf.pow((net[content_layer] - content_features[content_layer]), 2))
            factor = 1. / (number * height * width)
            content_losses.append(content_layers_weights[content_layer] * content_weight * gram_diff * factor)
        content_loss += reduce(tf.add, content_losses)

        # style loss
        style_loss = 0
        for i in range(len(styles)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                _, height, width, number = map(lambda j: j.value, layer.get_shape())
                feats = tf.reshape(layer, (-1, number))
                gram = tf.matmul(tf.transpose(feats), feats) / (height * width * number)
                style_gram = style_features[i][style_layer]
                gram_diff = tf.reduce_sum(tf.pow((gram - style_gram), 2))
                style_losses.append(style_layers_weights[style_layer] * gram_diff / style_gram.size)
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)

        # total variation denoising
        tv_y_size = _tensor_size(image[:, 1:, :, :])
        tv_x_size = _tensor_size(image[:, :, 1:, :])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1]-1, :, :]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2]-1, :]) /
                    tv_x_size))
        # overall loss
        loss = content_loss + style_loss + tv_loss

        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)
        
        ##########################################################################################
        # Thanks Anish Athalye for the optimizer setup
        #https://github.com/anishathalye/neural-style

        def print_progress():
            stderr.write('  content loss: %g\n' % content_loss.eval())
            stderr.write('    style loss: %g\n' % style_loss.eval())
            stderr.write('       tv loss: %g\n' % tv_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')
            if print_iterations and print_iterations != 0 > 0:
                print_progress()
            for i in range(iterations):
                stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                train_step.run()

                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    print_progress()

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = ih.postprocess(best.reshape(shape[1:]))

                    if preserve_colors:
                        original_image = np.clip(content, 0, 255)
                        styled_image = np.clip(img_out, 0, 255)

                        # Luminosity transfer steps:
                        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                        # 2. Convert stylized grayscale into YUV (YCbCr)
                        # 3. Convert original image into YUV (YCbCr)
                        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                        # 5. Convert recombined image from YUV back to RGB

                        # 1
                        styled_grayscale = ih.rgb2gray(styled_image)
                        styled_grayscale_rgb = ih.gray2rgb(styled_grayscale)

                        # 2
                        styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                        # 3
                        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                        # 4
                        w, h, _ = original_image.shape
                        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                        combined_yuv[..., 1] = original_yuv[..., 1]
                        combined_yuv[..., 2] = original_yuv[..., 2]

                        # 5
                        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))


                    yield (
                        (None if last_step else i),
                        img_out
                    )
        ##########################################################################################

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


