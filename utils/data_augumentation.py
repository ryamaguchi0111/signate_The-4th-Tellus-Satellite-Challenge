import numpy as np
import random

from PIL import ImageOps
import torch
from torchvision import transforms
from skimage.transform import resize


class Compose(object):
    """引数transformに格納された変形を順番に実行するクラス
       対象画像とアノテーション画像を同時に変換させます。 
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img

class RandomCrip(object):
    def __init__(self, cut_size):
        self.cut_size = cut_size
    def __call__(self, img, anno_class_img):
        width = img.shape[0]
        height = img.shape[1]

        start_w = random.randint(0, width - self.cut_size)
        start_h = random.randint(0, height - self.cut_size)

        img = img[start_w:start_w+self.cut_size, start_h:start_h+self.cut_size]
        anno_class_img = anno_class_img[start_w:start_w+self.cut_size, start_h:start_h+self.cut_size]

        return img, anno_class_img

class Mono2Color(object):
    def to8bit(self, img):
        img_8bit = img * (255/ (img.max()-img.min())) 
        img_8bit = np.uint8(img_8bit)
        return img_8bit
    def padding_median(self,image):
        img = image.copy()
        med = np.median(img)
    
        img[img==0] = med
        return img   
    def image_transform(self, img, mode):
        img = self.padding_median(img)
        if mode==0:
            img = self.to8bit(img)
        elif mode==1:
            img = np.log10(img+1e-2)
            img = self.to8bit(img)
        else:
            img_log = np.log10(img+1e-2)
            img_down = self.to8bit(img)
            img_down_log = self.to8bit(img_log)
            img = (img_down + img_down_log)/2
        return img       
    def mono2color(self, img):
        img_1ch = self.image_transform(img,0)
        img_2ch = self.image_transform(img,1)
        img_3ch = self.image_transform(img,2)

        return np.array([img_1ch, img_2ch, img_3ch], dtype=np.uint8)

    def __call__(self, img, anno_class_img):
        img = self.mono2color(img)
        img = img.transpose(1,2,0)
        return img, anno_class_img


class Resize(object):
    def __init__(self, rate):
        self.rate = rate
 
    def resize_img(self, img, rate):
       height = int(img.shape[0] * rate)
       width  = int(img.shape[1] * rate)
       img_resized = resize(img, (height, width))
       return img_resized   

    def __call__(self, img, anno_class_img):
        # img : numpy
        # anno_class_img : pillow
        img = self.resize_img(img, self.rate)
        anno_class_img = anno_class_img.resize(
            (int(anno_class_img.size[0]*self.rate),int(anno_class_img.size[1]*self.rate)),
             Image.NEAREST)
        anno_class_img = np.array(anno_class_img)
        return img, anno_class_img

class RandomMirror(object):
    """50%の確率で左右反転させるクラス"""

    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = np.fliplr(img)
            anno_class_img = np.fliplr(anno_class_img)
        return img, anno_class_img
			
class RandomRotation(object):
    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            i = np.random.randint(1,4)
            img = np.rot90(img, i)
            anno_class_img = np.rot90(anno_class_img, i)
        return img, anno_class_img


class Normalize_Tensor(object):
    def __init__(self, data_mean, data_std):
        self.data_mean = data_mean
        self.data_std = data_std

    def __call__(self, img, anno_class_img):
        # img を Tensor に変換
        img = transforms.functional.to_tensor(img.copy())

        # dataの標準化
        img = transforms.functional.normalize(
            img, self.data_mean, self.data_std
        )
        # annotation img を Tensor に変換
        anno_class_img = torch.from_numpy(anno_class_img.copy())

        return img, anno_class_img
