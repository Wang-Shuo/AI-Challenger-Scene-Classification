import torch
from torchvision import transforms
import shutil
from PIL import Image, ImageEnhance
import random
import numpy as np 
import os
from tqdm import tqdm


def adjust_learning_rate(optimizer, epoch, initial_lr):
    
    """set the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=1000, power=0.9):

    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    lr = init_lr * (1 - iter / max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', 'model_best.pth.tar')



class AverageMeter(object):
    """computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """computes the precision k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
        
    return res

# https://github.com/chplushsieh/carvana-challenge/blob/master/util/fancy_pca.py
# https://github.com/YinongLong/image-augmentation/blob/master/image_expandation.py

"""
rgb_mean = [126.07428161, 121.73672821, 114.10555781]

covar_mat:
[[ 5490.4303934   4822.21182581  4349.3670799 ]
 [ 4822.21182581  5315.46486994  5035.4421561 ]
 [ 4349.3670799   5035.4421561   5774.81094583]]

evals:
[ 15002.30776985   1281.80560554    296.59283377]

evecs:
[[-0.56332942 -0.73146231  0.38421719]
 [-0.58416557  0.0237431  -0.81128716]
 [-0.58430347  0.68146838  0.44067028]]

"""

def perform_pca(dataset_path):

    rgb_matrix = np.zeros(shape=(1,3))

    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for img_file in tqdm(filenames):
            img = Image.open(os.path.join(dirpath, img_file)).convert('RGB')
            img = img.resize((100,100))
            img_array = np.array(img).reshape(-1, 3)
            rgb_matrix = np.concatenate((rgb_matrix, img_array), axis=0)

    rgb_matrix = np.delete(rgb_matrix, (0), axis=0)
    print(rgb_matrix.shape)
    rgb_mean = rgb_matrix.mean(axis=0)
    print(rgb_mean)
    rgb_matrix = rgb_matrix - rgb_mean
    covar_mat = np.cov(rgb_matrix, rowvar=False)
    print(covar_mat)
    evals, evecs = np.linalg.eig(covar_mat)
    print(evals)
    print(evecs)


class fancy_pca(object):

    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, img):

        if random.random() < 0.5:
            evals = np.array([15002.3078, 1281.8056, 296.5928])
            evecs = np.array([[-0.56332942, -0.73146231,  0.38421719],
                              [-0.58416557,  0.0237431,  -0.81128716],
                              [-0.58430347,  0.68146838,  0.44067028]])

            feature_vec=np.matrix(evecs)

            # 3 x 1 scaled eigenvalue matrix
            se = np.zeros((3,1))
            se[0][0] = np.random.normal(self.mu, self.sigma)*evals[0]
            se[1][0] = np.random.normal(self.mu, self.sigma)*evals[1]
            se[2][0] = np.random.normal(self.mu, self.sigma)*evals[2]
            se = np.matrix(se)
            val = feature_vec*se

            img = np.array(img)
            img_pca=np.zeros((img.shape[0], img.shape[1], img.shape[2]))
            for k in range(img.shape[2]):
                img_pca[:,:,k]=np.matrix.__add__(img[:,:,k], val[k])
                img_pca[img_pca[:,:,k]>255]=255
                img_pca[img_pca[:,:,k]<0]=0

            return Image.fromarray(np.uint8(img_pca))

        return img 



#################################################################################
#https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py

def _adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        PIL Image: Brightness adjusted image.
    """

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def _adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        PIL Image: Contrast adjusted image.
    """

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def _adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        PIL Image: Saturation adjusted image.
    """

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def _adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See https://en.wikipedia.org/wiki/Hue for more details on Hue.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        PIL Image: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))


    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


# https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        tfs = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            tfs.append(transforms.Lambda(lambda img: _adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            tfs.append(transforms.Lambda(lambda img: _adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            tfs.append(transforms.Lambda(lambda img: _adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            tfs.append(transforms.Lambda(lambda img: _adjust_hue(img, hue_factor)))

        np.random.shuffle(tfs)
        transform = transforms.Compose(tfs)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """

        if random.random() < 0.5:
            transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
            return transform(img)

        return img



def _rotate(img, angle, resample=False, expand=False, center=None):
    """Rotate the image by angle and then (optionally) translate it by (n_columns, n_rows)
    """
    return img.rotate(angle, resample, expand, center)


# https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = np.random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return _rotate(img, angle, self.resample, self.expand, self.center)



if __name__ == '__main__':
    img = Image.open('test.jpg').convert('RGB')
    img.show()
    #dataset_path = '/home/wangshuo/compet/AIC/datasets/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922'
    #perform_pca(dataset_path)




