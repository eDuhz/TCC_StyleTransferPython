import scipy.misc
import numpy as np
import PIL.Image


MEAN_PIXEL = [103.939, 116.779, 123.68]

def rgb2gray(rgb):
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    return gray

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

def preprocess(img):
    return img - MEAN_PIXEL

def postprocess(img):
    return img + MEAN_PIXEL

def imread(path, crop=False, crop_ratio=0.5):

    img = scipy.misc.imread(path).astype(np.float)

    if crop:

        if not (0 <= img).all() and not (img <= 1.0).all():
            raise ValueError('Image not normalized')

        x, y, channel = img.shape
        # original image width and height cropped crop_ratio%
        xcrop, ycrop = x * crop_ratio, y * crop_ratio
        xoff = (x - xcrop) // 2
        yoff = (y - ycrop) // 2
        crop_img = img[xoff:-xoff, yoff:-yoff]

        return crop_img
    else:

        if len(img.shape) == 2:
            # grayscale
            img = np.dstack((img, img, img))
        elif img.shape[2] == 4:
            # PNG with alpha channel
            img = img[:, :, :3]

        return img

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    PIL.Image.fromarray(img).save(path, quality=95)

def imresize(image, width, kind='content', scale=1.0):
    if kind == 'content':
        content_image = image
        new_shape = (int(content_image.shape[0]//content_image[1] * width), width)
        content_image = scipy.misc.imresize(content_image, new_shape)
        return content_image
    elif kind == 'style':
        style_image = image
        style_image = scipy.misc.imresize(style_image, scale*width/style_image.shape[1])
        return style_image
    elif kind == 'initial':
        initial_image = image
        return scipy.misc.imresize(initial_image, width)
