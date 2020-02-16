import cv2
import torch
import numpy as np
from copy import copy
import glob
from PIL import Image
from torch.nn import LocalResponseNorm
import torch.nn.functional as func
import torchvision.transforms.functional as F
from torchvision import transforms

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None, mask=None):
        for t in self.transforms:
            image, target, mask = t(image, target, mask)
        return image, target, mask


class OneOf(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target=None, mask=None):
        transform = np.random.choice(self.transforms)
        image, target, mask = transform(image, target, mask)
        return image, target, mask
    
    
class RandomApply(object):
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob    
        
    def __call__(self, image, target=None, mask=None):
        for t in self.transforms:
            if np.random.rand() < self.prob:
                image, target, mask = t(image, target, mask=None)
        return image, target, mask


class RandomApply(object):
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob
        
    def __call__(self, image, target=None, mask=None):
        for t in self.transforms:
            if np.random.rand() < self.prob:
                image, target, mask = t(image, target, mask)
        return image, target, mask


class CenterCrop(object):
    def __call__(self, image, target=None, mask=None):
        height, width = image.shape[:2]
        
        if height > width:
            center = height // 2
            top = center - width // 2
            bottom = center + width // 2
            image = image[top:bottom, :]
            
            if target is not None:
                target[:, 1:-1:2] -= (height - width) // 2
        else:
            center = width // 2
            left = center - height // 2
            right = center + height // 2
            image = image[:, left:right]
            
            if target is not None:
                target[:, 0:-1:2] -= (width - height) // 2
            
        return image, target, mask


    
class Distort(object):        
    def __call__(self, image, target=None, mask=None):
        image = image.copy()
    
        if np.random.randint(2):
    
            #brightness distortion
            if np.random.randint(2):
                self._convert(image, beta=np.random.uniform(-32, 32))
    
            #contrast distortion
            if np.random.randint(2):
                self._convert(image, alpha=np.random.uniform(0.5, 1.5))
    
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
            #saturation distortion
            if np.random.randint(2):
                self._convert(image[:, :, 1], alpha=np.random.uniform(0.5, 1.5))
    
            #hue distortion
            if np.random.randint(2):
                tmp = image[:, :, 0].astype(int) + np.random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp
    
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    
        else:
    
            #brightness distortion
            if np.random.randint(2):
                self._convert(image, beta=np.random.uniform(-32, 32))
    
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
            #saturation distortion
            if np.random.randint(2):
                self._convert(image[:, :, 1], alpha=np.random.uniform(0.5, 1.5))
    
            #hue distortion
            if np.random.randint(2):
                tmp = image[:, :, 0].astype(int) + np.random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp
    
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    
            #contrast distortion
            if np.random.randint(2):
                self._convert(image, alpha=np.random.uniform(0.5, 1.5))
    
        return image, target, mask
    
    def _convert(self, image, alpha=1, beta=0):
        tmp = image.astype(np.float32) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

        
class GrayScale(object):
    def __call__(self, image, target=None, mask=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image, target, mask        


class Contrast(object):
    def __init__(self, lower=0.9, upper=1.1):
        self.lower = lower
        self.upper = upper

    def __call__(self, image, target=None, mask=None):
        alpha = np.random.uniform(self.lower, self.upper)
        image = image.astype(np.float)
        image *= alpha
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        return image, target, mask


class Brightness(object):
    def __init__(self, delta=16):
        self.delta = delta

    def __call__(self, image, target=None, mask=None):
        delta = np.random.randint(-self.delta, self.delta)
        image = image.astype(np.int)
        image += delta
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        return image, target, mask
    

class GaussianBlur(object):
    def __init__(self, kernel=3):
        self.kernel = (kernel, kernel)
    
    def __call__(self, image, target=None, mask=None):
        image = cv2.blur(image, self.kernel)
        return image, target, mask


class Expand(object):
    def __init__(self, size=1024, diff=0.3, noise=False):
        self.size = size
        self.noise = noise
        self.diff = diff

    def __call__(self, image, target=None, mask=None):
        height, width = image.shape[:2]
        max_ratio = self.size / max(height, width)
        min_ratio = max_ratio * self.diff

        ratio = np.random.uniform(min_ratio, max_ratio)
        left = np.random.uniform(0, self.size - width*ratio)
        top = np.random.uniform(0, self.size - height*ratio)

        expand_image = np.zeros((self.size, self.size, 3), dtype=image.dtype)
        if self.noise:
            mean = np.full(3, 0.5)
            std = np.full(3, 0.5)
            expand_image = cv2.randn(expand_image, mean, std)
        expand_image = np.clip(expand_image, 0, 1)
        
        image = cv2.resize(image, (int(width*ratio), int(height*ratio)))
        
        expand_image[int(top):int(top) + int(height*ratio),
                     int(left):int(left) + int(width*ratio)] = image
        image = expand_image
        
        if target is not None:
            target[:, 0:-1:2] = target[:, 0:-1:2] * ratio + left
            target[:, 1:-1:2] = target[:, 1:-1:2] * ratio + top

        return image, target, mask


class Pad(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image, target=None, mask=None):
        height, width = image.shape[:2]
        
        ratio = self.size / max(height, width)
        
        new_height = int(height * ratio)
        new_width = int(width * ratio)
        
        # new_size should be in (width, height) format
        
        image = cv2.resize(image, (new_width, new_height))
        
        delta_w = self.size - new_width
        delta_h = self.size - new_height
        
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        if target is not None:
            target[:, 0:-1:2] = target[:, 0:-1:2] * ratio + left
            target[:, 1:-1:2] = target[:, 1:-1:2] * ratio + top
        
        color = [0, 0, 0]
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        return image, target, mask
    
    
class Rotate(object):
    def __init__(self, angle=10, aligne=False):
        self.angle = angle
        self.aligne = aligne
        
    def __call__(self, image, target=None, mask=None):        
        angle = np.random.uniform(-self.angle, self.angle)

        height, width = image.shape[:2]
        cX, cY = width / 2, height / 2
     
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

        if self.aligne:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
     
            width = int((height * sin) + (width * cos))
            height = int((height * cos) + (width * sin))
     
            M[0, 2] += (width / 2) - cX
            M[1, 2] += (height / 2) - cY
            
        if target is not None:
            target = self._transform_landmarks(target, M)
            target = self._transform_boxes(target, M)
            
        image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_CONSTANT)
        return image, target, mask
    
    def _transform_landmarks(self, target, M):
        points_num = target[0, 4:-1].size
        points = np.ones((target[:, 4:-1].size // 2, 3))
        points[:, :2] = target[:, 4:-1].reshape(-1, 2)
        points = np.dot(points, M.T)
        
        target[:, 4:-1] = points.reshape(-1, points_num)
        return target
    
    def _transform_boxes(self, target, M):
        x1 = target[:, 0]
        y1 = target[:, 1]
        
        x2 = target[:, 0]
        y2 = target[:, 3]
        
        x3 = target[:, 2]
        y3 = target[:, 1]
        
        x4 = target[:, 2]
        y4 = target[:, 3]
        
        corners = np.stack([x1,y1,x2,y2,x3,y3,x4,y4]).transpose()
        
        points = np.ones((corners.size // 2, 3))
        points[:, :2] = corners.reshape(-1, 2)
        points = np.dot(points, M.T)
        
        points = points.reshape(-1, 8)
        
        x_ = points[:, 0::2]
        y_ = points[:, 1::2]
        
        xmin = np.min(x_, 1).reshape(-1,1)
        ymin = np.min(y_, 1).reshape(-1,1)
        xmax = np.max(x_, 1).reshape(-1,1)
        ymax = np.max(y_, 1).reshape(-1,1)
    
        target[:, :4] = np.hstack((xmin, ymin, xmax, ymax)).reshape(-1, 4)
        return target
    

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None, mask=None):
        height, width = image.shape[:2]
        
        h_scale = self.size / height
        w_scale = self.size / width
        
        if target is not None:
            target[:, 1:-1:2] *= h_scale
            target[:, 0:-1:2] *= w_scale
        
        image = cv2.resize(image, (self.size, self.size))
        return image, target, mask


class HorizontalFlip(object):
    def __call__(self, image, target=None, mask=None):
        height, width = image.shape[:2]
        
        if target is not None:
            
            x1 = target[:, 0].copy() 
            x2 = target[:, 2].copy()
            target[:, 0] = width - x2
            target[:, 2] = width - x1
            
            x1 = target[:, 4].copy()
            y1 = target[:, 5].copy()
            x2 = target[:, 6].copy()
            y2 = target[:, 7].copy()
            x3 = target[:, 8].copy()
            y3 = target[:, 9].copy()
            x4 = target[:, 10].copy()
            y4 = target[:, 11].copy()
            x5 = target[:, 12].copy()
            y5 = target[:, 13].copy()
            target[:, 4] = width - x2
            target[:, 5] = y2
            target[:, 6] = width - x1
            target[:, 7] = y1
            target[:, 8] = width - x3
            target[:, 9] = y3
            target[:, 10] = width - x5
            target[:, 11] = y5
            target[:, 12] = width - x4
            target[:, 13] = y4
            
            mask[:, 4:6], mask[:, 6:8] = mask[:, 6:8].copy(), mask[:, 4:6].copy()
            mask[:, 10:12], mask[:, 12:14] = mask[:, 12:14].copy(), mask[:, 10:12].copy()
        
        image = cv2.flip(image, 1)
        return image, target, mask


class LocalRespNorm(object):
    """
    Local Response Normalisation.
    https://medium.com/@dibyadas/visualizing-different-normalization-techniques-84ea5cc8c378
    This class is wrapper over torch.nn.LocalResponseNorm
    """
    def __init__(self, size=3, alpha=1e-4, beta=0.75, k=1):
        """
        Init parameters:
        :param size: (int), amount of neighbouring channels used for normalization (kernel size)
        :param alpha: (float), multiplicative factor. Default: 0.0001
        :param beta: (float), exponent. Default: 0.75
        :param k: (float), additive factor. Default: 1
        """
        self.LocalResponseNorm = LocalResponseNorm(size, alpha, beta, k)

    def __call__(self, image, target=None, mask=None):
        if len(image.shape) < 3:
            image = np.expand_dims(image, 2)
#         real_image = image.copy()

        if len(image.shape) > 2:
            image_tensor = torch.from_numpy(np.expand_dims(image.transpose((2, 0, 1)), 0)).to(torch.float32)
        else:
            image_tensor = torch.from_numpy(np.expand_dims(image.transpose((1, 0)), 0)).to(torch.float32)

        lrn_image = self.LocalResponseNorm(image_tensor)
        transformed_img = lrn_image[0].numpy().transpose((1, 2, 0))

        numerator = (transformed_img - transformed_img.min())
        denominator = (transformed_img.max() - transformed_img.min()) + 1e-6  # to avoid zero division
        scaled_transformed_img = numerator / denominator

#         if scaled_transformed_img.shape[2] == 1:
#             scaled_transformed_img = np.squeeze(scaled_transformed_img)
#         if len(scaled_transformed_img.shape) > 2:
#             scaled_transformed_img = cv2.cvtColor(scaled_transformed_img, cv2.COLOR_BGR2GRAY)
#         sample['real_image'] = real_image
        return scaled_transformed_img, target, mask


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    # def __init__(self, with_angles=False):
    #     self.with_angles = with_angles

    def __call__(self, image, target=None, mask=None):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        # if not self.with_angles:
        #     gaze = gaze[:-1]
        height, width = image.shape[:2]
#         print(image.shape)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
#         print(image)
        
        if target is not None:
            target[:, 1:-1:2] /= height
            target[:, 0:-1:2] /= width
            target = torch.from_numpy(target)
            
        if mask is not None:
            mask = torch.from_numpy(mask.astype(np.bool))
        
        return image/255., target, mask

class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor, target=None, mask=None):
        return F.normalize(tensor, self.mean, self.std, self.inplace), target, mask
    


# class ToTensor(object):
#     def __call__(self, image, target=None, mask=None):
#         height, width = image.shape[:2]
        
#         image = (image / 255).astype(np.float32)
#         image = image.transpose((2, 0, 1))
#         image = torch.from_numpy(image)
        
#         if target is not None:
#             target[:, 1:-1:2] /= height
#             target[:, 0:-1:2] /= width
#             target = torch.from_numpy(target)
            
#         if mask is not None:
#             mask = torch.from_numpy(mask.astype(np.uint8))
        
#         return image, target, mask


class Transforms(object):
    def __init__(self, input_size=1024, train=True):
        self.train = train
        self.transforms = RandomApply([
        ])

        self.normalize = Compose([
            Pad(size=input_size),
            ToTensor(),
        ])

    def __call__(self, image, target=None, mask=None):
        if self.train:
            image, target, mask = self.transforms(image, target, mask)
        image, target, mask = self.normalize(image, target, mask)
        return image, target, mask
