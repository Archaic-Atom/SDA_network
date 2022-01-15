# -*- coding: utf-8 -*-
from JackBasicStructLib.Basic.Define import *
from .ImgHandler import *


def RandomOrg(w, h, crop_w, crop_h):
    x = random.randint(0, w - crop_w)
    y = random.randint(0, h - crop_h)
    return x, y


def VerticalFlip(imgL, imgR, disp_gt, cls_gt):
    flip_prop = np.random.randint(low=0, high=2)

    if flip_prop == 0:
        imgL = cv2.flip(imgL, 0)
        imgR = cv2.flip(imgR, 0)
        disp_gt = cv2.flip(disp_gt, 0)
        # cls_gt = cv2.flip(cls_gt, 0)

    cls_gt = None
    return imgL, imgR, disp_gt, cls_gt


def HorizontalFlip(img, imgGround, axis):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=3)
    if flip_prop == 0:
        img = cv2.flip(img, 0)
        imgGround = cv2.flip(imgGround, 0)
    elif flip_prop == 1:
        img = cv2.flip(img, 1)
        imgGround = cv2.flip(imgGround, 1)

    flip_prop = np.random.randint(low=0, high=3)
    if flip_prop == 0:
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)
        imgGround = cv2.transpose(imgGround)
        imgGround = cv2.flip(imgGround, 0)
    elif flip_prop == 1:
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)
        imgGround = cv2.transpose(imgGround)
        imgGround = cv2.flip(imgGround, 1)

    return img, imgGround


def NormalizeRGB(img):
    img = img.astype(float)
    for i in range(IMG_DEPTH):
        minval = img[:, :, i].min()
        maxval = img[:, :, i].max()
        if minval != maxval:
            img[:, :, i] = (img[:, :, i]-minval)/(maxval-minval)
    return img


def Standardization(img):
    """ normalize image input """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0, 1), keepdims=True)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + EPSILON)


def ImgProcessing(img):
    img = NormalizeRGB(img)
    img = img[:, :, :]
    return img


# Slice
def ImgSlice(img, x, y, w, h):
    return img[y:y+h, x:x+w, :]


def ImgGroundSlice(img, x, y, w, h):
    return img[y:y+h, x:x+w]


def RandomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def TestDataAugmentation(img):
    img90 = np.array(np.rot90(img))
    img1 = np.concatenate([img[None], img90[None]])
    img2 = np.array(img1)[:, ::-1]  # h_ v
    img3 = np.concatenate([img1, img2])
    img4 = np.array(img3)[:, :, ::-1]  # w_ v
    img5 = np.concatenate([img3, img4])
    return img5


def StyleDataAugmentation(imgL, imgR, imgRef):
    flip_prop = np.random.randint(low=0, high=10)
    if flip_prop <= -2:
        imgL = StyleTransfer(imgL, imgRef)
        imgR = StyleTransfer(imgR, imgRef)
    elif flip_prop <= 10:
        imgL = RGB2GRAY(imgL)
        imgR = RGB2GRAY(imgL)

    return imgL, imgR


def DispDataAugmentation():
    flip_prop = np.random.randint(low=0, high=10)
    d = 0
    if flip_prop <= 4:
        d = np.random.randint(low=1, high=20)

    return d


def AdjustBrightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    if not IsNumpyImage(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([i*brightness_factor for i in range(0, 256)]).clip(0, 255).astype('uint8')
    # same thing but a bit slower
    # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    if img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)


def AdjustContrast(img, contrast_factor):
    """Adjust contrast of an mage.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    # much faster to use the LUT construction than anything else I've tried
    # it's because you have to change dtypes multiple times
    if not IsNumpyImage(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([(i-74)*contrast_factor+74 for i in range(0, 256)]
                     ).clip(0, 255).astype('uint8')
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(contrast_factor)
    if img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)


def AdjustGamma(img, gamma, gain=1):
    r"""Perform gamma correction on an image.
    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:
    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}
    See `Gamma Correction`_ for more details.
    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.
    """
    if not IsNumpyImage(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')
    # from here
    # https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python/41061351
    table = np.array([((i / 255.0) ** gamma) * 255 * gain
                      for i in np.arange(0, 256)]).astype('uint8')
    if img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)


def ChromaticTransformations(img, brightness_factor, gamma, contrast_factor):
    img = AdjustBrightness(img, brightness_factor)
    img = AdjustGamma(img, gamma)
    img = AdjustContrast(img, contrast_factor)
    return img


def Crop(img, sar, x, y, img_crop_size, sar_crop_size):

    img_size = img.shape[0]
    sar_size = sar.shape[0]

    def NewSarCdn():
        crop_sar_x = np.random.randint(0,  sar_size - sar_crop_size)
        crop_sar_y = np.random.randint(0,  sar_size - sar_crop_size)
        return crop_sar_x, crop_sar_y

    def NewXy(new_sar_x, new_sar_y, x, y):
        new_x = random.randrange(max(0, img_crop_size-img_size+new_sar_x+x),
                                 min(new_sar_x+x, img_crop_size-sar_crop_size))
        new_y = random.randint(max(0, (img_crop_size-img_size+new_sar_y+y)),
                               min((new_sar_y+y), (img_crop_size-sar_crop_size)))
        return new_x, new_y

    new_sar_x, new_sar_y = NewSarCdn()
    # new_img_x, new_img_y = new_img_cdn(new_sar_x, new_sar_y, x, y)
    new_x, new_y = NewXy(new_sar_x, new_sar_y, x, y)
    new_img_x = new_sar_x + x - new_x
    new_img_y = new_sar_y + y - new_y
    new_sar = sar[new_sar_x:new_sar_x + sar_crop_size, new_sar_y:new_sar_y + sar_crop_size]
    new_img = img[new_img_x:new_img_x + img_crop_size, new_img_y:new_img_y + img_crop_size]
    # new_x = x + new_sar_x - new_img_x
    # new_y = y + new_sar_y - new_img_y
    return new_img, new_sar, new_x, new_y


def RandomRotateFlip(img, sar, x, y,  img_crop_size, sar_crop_size):
    num = random.choice((-1, 1))   #1表示逆时针， -1表示顺时针
    k = random.choice((1,2,3,4))
    k_num = num * k
    choice_H = random.choice((0, 1))
    choice_W = random.choice((0, 1))
    new_sar = np.rot90(sar, k_num)
    new_img = np.rot90(img, k_num)
    if num == -1:
        if k == 1:
            new_x = img_crop_size - sar_crop_size - y
            new_y = x
        elif k == 2:
            new_x = img_crop_size - sar_crop_size - x
            new_y = img_crop_size - sar_crop_size - y
        elif k == 3:
            new_x = y
            new_y = img_crop_size -sar_crop_size - x
        elif k == 4:
            new_x = x
            new_y = y
        else:
            print("k:{}".format(k))
            print("error")
        
    elif num == 1:
        if k == 1:
            new_x = y
            new_y = img_crop_size - sar_crop_size - x
        elif k == 2:
            new_x = img_crop_size - sar_crop_size - x
            new_y = img_crop_size - sar_crop_size - y
        elif k == 3:
            new_x = img_crop_size -sar_crop_size - y
            new_y = x
        elif k == 4:
            new_x = x
            new_y = y
        else:
            print("k:{}".format(k))
            print("error")

    if choice_H == 1:
        new_sar = np.flip(new_sar, 0)
        new_img = np.flip(new_img, 0)
        new_x = new_x
        new_y = img_crop_size - sar_crop_size - new_y
    elif choice_H == 0:
        new_sar = new_sar
        new_img = new_img
        new_x = new_x
        new_y = new_y
    else:
        print("chioce_H:{}".format(choice_H))
        print("error")
    

    if choice_W == 1:
        new_sar = np.flip(new_sar, 1)
        new_img = np.flip(new_img, 1)
        new_x = img_crop_size - sar_crop_size -new_x
        new_y = new_y
    elif choice_W == 0:
        new_sar = new_sar
        new_img = new_img
        new_x = new_x
        new_y = new_y
    else:
        print("chioce_W:{}".format(choice_W))
        print("error")

    return  new_img, new_sar,  new_x, new_y