# -*- coding:UTF-8 -*-
import cv2
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

##每个通道减一个固定值
class SubtractMeans(object):
    def __init__(self,mean):
        self.mean = np.array(mean,dtype=np.float32)

    def __call__(self, image,boxes=None,labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return  image.astype(np.float32),boxes,labels

#将图像resize到300*300
class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels

##从真实像素框变回百分比的框
class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


##镜像反转
class RandomMirror(object):
    def __call__(self,image,boxes,classes):
        _,width,_=image.shape
        if random.randint(2):
            image = image[:,::-1]
            boxes = boxes.copy()
            boxes[:,0::2] = width-boxes[:,2::-2]

        return image,boxes,classes


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

##求IOU
def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union

class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            None,
            (0.1,None),
            (0.3,None),
            (0.7,None),
            (0.9,None),
            (None,None),
        )

    def __call__(self,image,boxes=None,labels=None):
        height,width,_ =image.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return image,boxes,labels
            min_iou,max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                if h / w < 0.5 or h / w > 2:
                        continue


                left = random.uniform(width - w)
                top = random.uniform(height - h)


                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                overlap = jaccard_numpy(boxes, rect)

                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],:]

                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2

                if not mask.any():
                    continue

                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]


                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],rect[2:])
                current_boxes[:, 2:] -= rect[:2]
                return current_image, current_boxes, current_labels





####把图像扩大,扩大的地方用0填充了
class Expand(object):
    def __init__(self,mean):
        self.mean = mean

    def __call__(self, image,boxes,labels):
        if random.randint(2):
            return image,boxes,labels

        height,width,depth = image.shape
        ratio = random.uniform(1,4)
        left = random.uniform(0,width*ratio-width)
        top = random.uniform(0,height*ratio-height)

        expand_image = np.zeros(
            (int(height*ratio),int(width*ratio),depth),
            dtype=image.dtype)
        expand_image[:,:,:] = self.mean
        expand_image[int(top):int(top+height),int(left):int(left+width)]=image

        image = expand_image

        boxes = boxes.copy()
        boxes[:,:2] += (int(left),int(top))
        boxes[:,2:] += (int(left),int(top))

        return image,boxes,labels

##在某范围随机调整图片色相
class RandomHue(object):
    def __init__(self,delta=18.0):
        assert delta>=0.0 and delta <=360.0
        self.delta = delta
    def __call__(self, image,boxes=None,labels=None):
        if random.randint(2):
            image[:,:,0] += random.uniform(-self.delta,self.delta)
            image[:,:,0][image[:,:,0]>360.0] -= 360.0
            image[:,:,0][image[:,:,0]<0.0] += 360.0
        return image,boxes,labels



#其中一个通道的值乘以一个值,改变饱和度
class RandomSaturation(object):
    def __init__(self,lower=0.5,upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower
        assert self.lower >= 0

    def __call__(self, image,boxes=None,labels=None):
        if random.randint(2):
            image[:,:,1] *= random.uniform(self.lower,self.upper)

        return image,boxes,labels


##转换图像颜色格式
class ConvertColor(object):
    def __init__(self,current='BGR',transform='HSV'):
        self.transform = transform
        self.current = current
    def __call__(self, image,boxes=None,labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)

        return image,boxes,labels

#所有像素乘一个值 改变对比度
class RandomContrast(object):
    def __init__(self,lower=0.5,upper=1.5):
        self.lower = lower
        self.upper = upper
        assert  self.upper >=self.lower
        assert  self.lower>=0

    def __call__(self, image,boxes=None,labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower,self.upper)
            image *= alpha
        return image,boxes,labels




#交换通道
class SwapChannels(object):
    def __init__(self,swaps):
        self.swaps = swaps

    def __call__(self, image):

        image = image[:,:,self.swaps]
        return image

##随机交换通道
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0,1,2),(0,2,1),
                      (1,0,2),(1,2,0),
                      (2,0,1),(2,1,0))

    def __call__(self,image,boxes=None,labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        return image,boxes,labels


## 相当于给像素值加一个噪声
class RandomBrightness(object):
    def __init__(self,delta=32):
        assert delta>=0.0
        assert delta<=255.0
        self.delta = delta
    def __call__(self,image,boxes=None,labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta,self.delta)
            image += delta

        return image,boxes,labels


##改变图像的对比度 饱和度 以及加一点噪声
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV',transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()


    def __call__(self, image,boxes,labels):
        im = image.copy()
        im,boxes,labels = self.rand_brightness(im,boxes,labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im,boxes,labels = distort(im,boxes,labels)

        return  self.rand_light_noise(im,boxes,labels)


##将百分比的boxes框转换为真实像素坐标
class ToAbsoluteCoords(object):
    def __call__(self,image,boxes=None,labels=None):
        height,width,channels = image.shape
        boxes[:,0] *=width
        boxes[:,2] *=width
        boxes[:,1] *=height
        boxes[:,3] *=height

        return image,boxes,labels


##将图像从int矩阵转为float矩阵。
class ConverFromInts(object):
    def __call__(self,image,boxes=None,labels=None):
        return image.astype(np.float32),boxes,labels



class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self,img,boxes=None,labels=None):
        for t in self.transforms:
            img,boxes,labels = t(img,boxes,labels)

        return img,boxes,labels



class SSDAugmentation(object):
    def __init__(self,size=300,mean=(104,117,123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConverFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)

        ])

    def __call__(self, img,boxes,labels):
        return self.augment(img,boxes,labels)

















