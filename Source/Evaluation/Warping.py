# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Copyright 2017 Modifications Clement Godard.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import tensorflow as tf
import tensorflow.contrib.slim as slim


def BilinearSamplerLDH(input_images, x_offset, wrap_mode='border',
                       name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):

            # handle both texture border types
            _edge_size = 0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            x = tf.clip_by_value(x, 0.0,  _width_f - 1 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_l = tf.gather(im_flat, idx_l)
            pix_r = tf.gather(im_flat, idx_r)

            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)

            return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                   tf.linspace(0.0, _height_f - 1.0, _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
            return output

    with tf.variable_scope(name):
        _num_batch = tf.shape(input_images)[0]
        _height = tf.shape(input_images)[1]
        _width = tf.shape(input_images)[2]
        _num_channels = tf.shape(input_images)[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f = tf.cast(_width,  tf.float32)

        _wrap_mode = wrap_mode

        output = _transform(input_images, x_offset)
        return output


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


def Smooth(img):
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    gy = img[:, :-1, :, :] - img[:, 1:, :, :]
    return [gx] + [gy]


def Warping_Loss(imgL, imgR, dsp):
    dsp = tf.expand_dims(dsp, axis=3)
    _, _, w, _ = imgL.get_shape().as_list()
    gen_imgL = BilinearSamplerLDH(imgR, -dsp / w)
    mask = BilinearSamplerLDH(dsp, -dsp / w)
    mask = tf.where(mask > 0, tf.ones_like(mask), tf.zeros_like(mask))
    imgL = imgL * mask

    # loss_1
    l1_loss = tf.reduce_mean(tf.abs(gen_imgL - imgL))
    ssim_loss = tf.reduce_mean(mask[:, 1:-1, 1:-1, :] * SSIM(imgL, gen_imgL))

    return l1_loss * 0.5 + ssim_loss * 0.8


if __name__ == "__main__":
    a = tf.placeholder(tf.float32, shape=(1, None, None, 3))    # img
    b = tf.placeholder(tf.float32, shape=(1, None, None, 1))  # disparty

    c = BilinearSamplerLDH(a, - b / 512.0)
    mask = BilinearSamplerLDH(b, -b / 512.0)
    mask = tf.where(mask > 0, tf.ones_like(mask), tf.zeros_like(mask))
    #c = mask * a

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    from PIL import Image
    import numpy as np
    import cv2
    import random

    def ImgSlice(img, x, y, w, h):
        return img[y:y+h, x:x+w, :]

    def RandomOrg(w, h, crop_w, crop_h):
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        return x, y

    path = "/Users/rhc/000000_10_L.png"

    imgL = Image.open(path).convert("RGB")
    imgL = np.array(imgL)
    w = 512
    h = 256
    x, y = RandomOrg(imgL.shape[1], imgL.shape[0], w, h)

    x = 150
    y = 150

    imgL = ImgSlice(imgL, x, y, w, h)
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    path = '000000_10.png'
    cv2.imwrite(path, imgL)
    imgL = np.expand_dims(imgL, axis=0)

    d = 200

    path = "/Users/rhc/000000_10_R.png"
    imgR = Image.open(path).convert("RGB")
    imgR = np.array(imgR)
    imgR = ImgSlice(imgR, x + d, y, w, h)
    path = '000001_10.png'
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, imgR)
    imgR = np.expand_dims(imgR, axis=0)

    path = "/Users/rhc/000000_10.png"
    imgGround = Image.open(path)
    imgGround = np.ascontiguousarray(imgGround, dtype=np.float32)/float(256.0)
    imgGround = imgGround + d
    imgGround = np.expand_dims(imgGround, axis=2)
    imgGround = ImgSlice(imgGround, x, y, w, h)
    imgGround = np.expand_dims(imgGround, axis=0)

    warping_img = sess.run(c, feed_dict={a: imgR, b: imgGround})

    img = np.array(warping_img)
    img = img[0]
    #img = (img * float(256.0)).astype(np.uint16)
    img = img.astype(np.uint8)
    path = "000002_10.png"
    print(img.shape)
    imgL = imgL[0]
    img = np.concatenate((imgL, img), axis=0)

    for i in range(50):
        w = i * 10
        img[:, w, 0] = 255
        img[:, w, 1] = 0
        img[:, w, 2] = 0

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)
