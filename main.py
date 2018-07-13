import os
import image_handler as ih
from style_transfer import stylize
from argparse import ArgumentParser


VGG_PATH = 'imagenet-vgg-verydeep-19.mat'

def blend_weights(style_images, style_blend_weights=None):
    if style_blend_weights is None:
        return [1.0/len(style_images) for _ in style_images]
    else:
        total_blend = sum(style_blend_weights, 0)
        blend = [weight for weight in style_blend_weights]
        blend /= total_blend
        return blend

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content', dest='content', help='content image', required=True)
    parser.add_argument('--styles', dest='styles', nargs='+', help='one or more style images', required=True)
    parser.add_argument('--output', dest='output', help='output path', required=True)
    parser.add_argument('--iterations', type=int, dest='iterations', help='iterations. Default = 1000',
                        default=1000)
    parser.add_argument('--print-iterations', type=int, dest='print_iterations', help='printing frequency')
    parser.add_argument('--checkpoint-output', dest='checkpoint_output',
                        help='checkpoint output format, e.g. output%%s.jpg')
    parser.add_argument('--checkpoint-iterations', type=int, dest='checkpoint_iterations', help='checkpoint frequency')
    parser.add_argument('--width', type=int, dest='width', help='output width', default=None)
    parser.add_argument('--style-scales', type=float, dest='style_scales', nargs='+', help='one or more style scales')
    parser.add_argument('--network', dest='network',
                        help='path to network parameters. Default = imagenet-vgg-verydeep-19.mat', default=VGG_PATH)
    parser.add_argument('--content-weight-blend', type=float, dest='content_weight_blend',
                        help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend), default = 1',
                        default=1)
    parser.add_argument('--content-weight', type=float, dest='content_weight',
                        help='content weight default=5e0', default=5e0)
    parser.add_argument('--style-weight', type=float, dest='style_weight', help='style weight, default=9e2',
                        default=1e4)
    parser.add_argument('--style-layer-weight-exp', type=float, dest='style_layer_weight_exp',
                        help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight('
                             'layer<n>). default=1',
                        default=1)
    parser.add_argument('--style-blend-weights', type=float, dest='style_blend_weights',
                        help='style blending weights, default = 1/n_styles',
                        nargs='+')
    parser.add_argument('--tv-weight', type=float, dest='tv_weight',
                        help='total variation regularization weight, default=1e2', default=1e2)
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', help='learning rate default=1e1',
                        default=1e1)
    parser.add_argument('--beta1', type=float, dest='beta1', help='Adam: beta1 parameter, default = 0.9',
                        default=0.9)
    parser.add_argument('--beta2', type=float, dest='beta2', help='Adam: beta2 parameter, default = 0.999',
                        default=0.999)
    parser.add_argument('--eps', type=float,
                        dest='epsilon', help='Adam: epsilon parameter, default = 1e-08', default=1e-08)
    parser.add_argument('--initial', dest='initial', help='initial image')
    parser.add_argument('--initial-noiseblend', type=float, dest='initial_noiseblend',
                        help='ratio of blending initial image with normalized noise')
    parser.add_argument('--preserve-colors', action='store_true', dest='preserve_colors', default=False,
                        help='style transfer without the style image colors.')
    parser.add_argument('--pooling', dest='pooling', default='max',
                        help='pooling layer configuration: max or avg. default = max')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.network):
        parser.error("Network not found!" % args.network)

    content_image = ih.imread(args.content)
    style_images = [ih.imread(style) for style in args.styles]

    width = args.width
    if width is not None:
        content_image = ih.imresize(content_image, width)

    target_shape = content_image.shape

    for i in range(len(style_images)):
        style_scale = 1.0
        if args.style_scales is not None:
            style_scale = args.style_scales[i]
        style_images[i] = ih.imresize(style_images[i], target_shape[1], 'style', style_scale)

    style_blend_weights = blend_weights(style_images, args.style_blend_weights)

    initial = content_image

    if args.initial is not None and args.initial_noiseblend is None:
        initial = ih.imresize(ih.imread(args.initial), content_image.shape[:2])
        args.initial_noiseblend = 0.0
    else:
        if args.initial_noiseblend is None:
            args.initial_noiseblend = 1.0

    ##########################################################################################
    # Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.
    if args.checkpoint_output and "%s" not in args.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

    for iteration, image in stylize(
            network=args.network,
            initial=initial,
            initial_noiseblend=args.initial_noiseblend,
            content=content_image,
            styles=style_images,
            preserve_colors=args.preserve_colors,
            iterations=args.iterations,
            content_weight=args.content_weight,
            content_weight_blend=args.content_weight_blend,
            style_weight=args.style_weight,
            style_layer_weight_exp=args.style_layer_weight_exp,
            style_blend_weights=style_blend_weights,
            tv_weight=args.tv_weight,
            learning_rate=args.learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            pooling=args.pooling,
            print_iterations=args.print_iterations,
            checkpoint_iterations=args.checkpoint_iterations
    ):
        output_file = None
        combined_rgb = image
        if iteration is not None:
            if args.checkpoint_output:
                output_file = args.checkpoint_output % iteration
        else:
            output_file = args.output
        if output_file:
            ih.imsave(output_file, combined_rgb)
    ##########################################################################################

if __name__ == '__main__':
    main()
