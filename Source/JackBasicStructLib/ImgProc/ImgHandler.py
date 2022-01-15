# -*- coding: utf-8 -*-
from JackBasicStructLib.Basic.Define import *


def ReadImg(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    return img


def ReadPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(b'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()

    return data, scale


def WritePFM(file, image, scale=1):
    file = open(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image_string = image.tostring()
    file.write(image_string)

    file.close()


def RGB2LAB(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img = img.astype(np.float32)
    img[:, :, 0] = img[:, :, 0] * 100 / 255
    img[:, :, 1] = img[:, :, 1] - 128
    img[:, :, 2] = img[:, :, 2] - 128
    return img


def LAB2RGB(img):
    img[:, :, 0] = img[:, :, 0] * 255 / 100
    img[:, :, 1] = img[:, :, 1] + 128
    img[:, :, 2] = img[:, :, 2] + 128
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img


def MeanVar(img):
    mean = np.mean(img)
    var = math.sqrt(np.var(img))
    return mean, var


def RGB2GRAY(img):
    imgG = copy.deepcopy(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgG[:, :, 0] = img
    imgG[:, :, 1] = img
    imgG[:, :, 2] = img
    return imgG


def StyleTransfer(imgS, ImgR):
    imgS_LAB = RGB2LAB(imgS)
    imgR_LAB = RGB2LAB(ImgR)

    mean_S, var_S = MeanVar(imgS_LAB)
    # print mean_S, var_S

    mean_R, var_R = MeanVar(imgR_LAB)
    # print mean_R, var_R

    imgS_LAB = imgS_LAB - mean_S
    lamda = var_R / var_S
    imgS_LAB = lamda * imgS_LAB
    imgS_LAB = imgS_LAB + mean_R

    imgS_RGB = LAB2RGB(imgS_LAB)

    return imgS_RGB


def IsNumpyImage(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def Sobel_Edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def AutoCanny(image, sigma=0.33):
    #计算单通道像素强度中位数
    v = np.median(image)

    #选择合适的Lower和Upper值，然后应用他们
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    
    return edged