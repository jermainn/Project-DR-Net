import numpy as np
import torch
from PIL import Image
import os


def mkdirs(paths):
    """
    创建多个新的文件夹
    :param paths: 文件夹路径
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """
    创建空文件夹
    :param path: (str) -- 路径
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)

def tensor2im(in_image, imtype=np.uint8):
    """
    将 tensor array 转换为 numpy array
    :param in_image: 输入图片
    :param imtype: 输出图片的数据类型
    :return:
    """

    if not isinstance(in_image, np.ndarray):
        if isinstance(in_image, torch.Tensor):
            image_tensor = in_image.data
        else:
            return in_image
        # 转换为 numpy 类型
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        image_numpy = image_tensor.cpu().numpy()
        if image_numpy.shape[0] == 1:       # 转换为 RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        if not isinstance(image_numpy, np.ndarray):
            # 下面那个加1是干什么
            image_numpy = (image_numpy.float() + 1) / 2.0 * 255
            # image_numpy = (image_numpy + 1) / 2.0 * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:
        image_numpy = in_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """
    保存图片
    :param image_numpy: 图片
    :param img_path: 路径
    :param aspect_ratio: H/W 的值，可选参数
    :return:
    """
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)

    image_pil.save(image_path)

