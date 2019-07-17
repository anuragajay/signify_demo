import os
import shutil
import time

import torch
import torch.utils.data
from . import imagenet_models as models
from torchvision import transforms, datasets
ch = torch

from . import constants
from . import loaders
from . import cifar_models
from . import folder

from .helpers import get_label_mapping

###
# Datasets: (all subclassed from dataset)
# In order:
## ImageNet
## Restricted Imagenet (+ Balanced)
## Other Datasets:
## - CIFAR
## - CINIC
## - A2B (orange2apple, horse2zebra, etc)
###

class DataSet(object):
    def __init__(self, ds_name):
        self.ds_name = ds_name
        self.num_classes = None
        self.mean = None
        self.std = None
        self.custom_class = None
        self.label_mapping = None

    def get_model(self, arch):
        raise RuntimeError('no get_model function!')

    def make_loaders(self, workers, batch_size, data_aug=True, subset=None, 
                     subset_start=0, subset_type='rand', val_batch_size=None,
                     only_val=False):
        transforms = (self.transform_train, self.transform_test)
        return loaders.make_loaders(workers=workers,
                                    batch_size=batch_size,
                                    transforms=transforms,
                                    data_path=self.data_path,
                                    data_aug=data_aug,
                                    dataset=self.ds_name,
                                    label_mapping=self.label_mapping,
                                    custom_class=self.custom_class,
                                    val_batch_size=val_batch_size,
                                    subset=subset,
                                    subset_start=subset_start,
                                    subset_type=subset_type,
                                    only_val=only_val)

class ImageNet(DataSet):
    def __init__(self, data_path, **kwargs):
        super(ImageNet, self).__init__('imagenet')
        self.data_path = data_path
        self.mean = constants.IMAGENET_MEAN
        self.std = constants.IMAGENET_STD
        self.num_classes = 1000

        self.transform_train = constants.TRAIN_TRANSFORMS_224
        self.transform_test = constants.TEST_TRANSFORMS_224

    def get_model(self, arch):
        return models.__dict__[arch](num_classes=self.num_classes)

class Places(DataSet):
    def __init__(self, data_path, **kwargs):
        super(Places, self).__init__('places')
        self.data_path = data_path
        self.mean = ch.tensor([0.0, 0.0, 0.0])
        self.std = ch.tensor([1.0, 1.0, 1.0])
        self.num_classes = 365

        self.transform_train = constants.TRAIN_TRANSFORMS_224
        self.transform_test = constants.TEST_TRANSFORMS_224

    def get_model(self, arch):
        return models.__dict__[arch](num_classes=self.num_classes)

class PlacesRoom(DataSet):
    def __init__(self, data_path, **kwargs):
        super(PlacesRoom, self).__init__('places_room')
        self.data_path = data_path
        self.mean = ch.tensor([0.0, 0.0, 0.0])
        self.std = ch.tensor([1.0, 1.0, 1.0])
        self.num_classes = 18

        self.transform_train = constants.TRAIN_TRANSFORMS_224
        self.transform_test = constants.TEST_TRANSFORMS_224

    def get_model(self, arch):
        return models.__dict__[arch](num_classes=self.num_classes)

class SunLamp(DataSet):
    def __init__(self, data_path, **kwargs):
        super(SunLamp, self).__init__('sun_lamp')
        self.data_path = data_path
        self.mean = ch.tensor([0.0, 0.0, 0.0])
        self.std = ch.tensor([1.0, 1.0, 1.0])
        self.num_classes = 2

        self.transform_train = constants.TRAIN_TRANSFORMS_224
        self.transform_test = constants.TEST_TRANSFORMS_224

    def get_model(self, arch):
        return models.__dict__[arch](num_classes=self.num_classes)

class PlacesFilterRoom(DataSet):
    def __init__(self, data_path, **kwargs):
        super(PlacesFilterRoom, self).__init__('places_filter_room')
        self.data_path = data_path
        self.mean = ch.tensor([0.0, 0.0, 0.0])
        self.std = ch.tensor([1.0, 1.0, 1.0])
        self.num_classes = 54

        self.transform_train = constants.TRAIN_TRANSFORMS_224
        self.transform_test = constants.TEST_TRANSFORMS_224

    def get_model(self, arch):
        return models.__dict__[arch](num_classes=self.num_classes)

class RestrictedImageNet(DataSet):
    def __init__(self, data_path, **kwargs):
        name = 'restricted_imagenet'
        super(RestrictedImageNet, self).__init__(name)
        self.data_path = data_path
        self.mean = constants.IMAGENET_MEAN
        self.std = constants.IMAGENET_STD
        self.num_classes = len(constants.RESTRICTED_RANGES)

        self.transform_train = constants.TRAIN_TRANSFORMS_224
        self.transform_test = constants.TEST_TRANSFORMS_224

        self.label_mapping = get_label_mapping(self.ds_name,
                constants.RESTRICTED_RANGES)

    def get_model(self, arch):
        return models.__dict__[arch](num_classes=self.num_classes)

class RestrictedImageNetBalanced(DataSet):
    def __init__(self, data_path, **kwargs):
        super(RestrictedImageNetBalanced, self).__init__('restricted_imagenet_balanced')
        self.data_path = data_path
        self.mean = constants.IMAGENET_MEAN
        self.std = constants.IMAGENET_STD
        self.num_classes = len(constants.BALANCED_RANGES)

        self.transform_train = constants.TRAIN_TRANSFORMS_224
        self.transform_test = constants.TEST_TRANSFORMS_224

        self.label_mapping = get_label_mapping(self.ds_name,
                constants.BALANCED_RANGES)

    def get_model(self, arch):
        return models.__dict__[arch](num_classes=self.num_classes)

class CIFAR(DataSet):
    def __init__(self, data_path='/tmp/', **kwargs):
        super(CIFAR, self).__init__('cifar')
        self.mean = constants.CIFAR_MEAN
        self.std = constants.CIFAR_STD
        self.num_classes = 10
        self.data_path = data_path

        self.transform_train = constants.TRAIN_TRANSFORMS(32)
        self.transform_test = constants.TEST_TRANSFORMS(32)

        self.custom_class = datasets.CIFAR10

    def get_model(self, arch):
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

class ROBUST_CIFAR(DataSet):
    def __init__(self, data_path='/tmp/', **kwargs):
        super(ROBUST_CIFAR, self).__init__('robust_cifar')
        self.mean = constants.CIFAR_MEAN
        self.std = constants.CIFAR_STD
        self.num_classes = 10
        self.data_path = data_path

        self.transform_train = constants.TRAIN_TRANSFORMS(32)
        self.transform_test = constants.TEST_TRANSFORMS(32)

        self.custom_class = cifar_helper

    def get_model(self, arch):
        return cifar_models.__dict__[arch](num_classes=self.num_classes)


class NON_ROBUST_CIFAR(DataSet):
    def __init__(self, data_path='/tmp/', **kwargs):
        super(NON_ROBUST_CIFAR, self).__init__('non_robust_cifar')
        self.mean = constants.CIFAR_MEAN
        self.std = constants.CIFAR_STD
        self.num_classes = 10
        self.data_path = data_path

        self.transform_train = constants.TRAIN_TRANSFORMS(32)
        self.transform_test = constants.TEST_TRANSFORMS(32)

        self.custom_class = cifar_helper

    def get_model(self, arch):
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

def cifar_helper(root, train, download, transform):
    data = ch.cat(ch.load(os.path.join(root, f"CIFAR_ims")))
    labels = ch.cat(ch.load(os.path.join(root, f"CIFAR_lab")))
    if train:
        data = data[:45000]
        labels = labels[:45000]
    else:
        data = data[45000:]
        labels = labels[45000:]
    dataset = folder.TensorDataset(data, labels, transform=transform)
    return dataset 

class CINIC(DataSet):
    def __init__(self, data_path, **kwargs):
        super(CINIC, self).__init__('cinic')
        self.data_path = data_path
        self.mean = constants.CINIC_MEAN
        self.std = constants.CINIC_STD
        self.num_classes = 10

        self.transform_train = constants.TRAIN_TRANSFORMS(32)
        self.transform_test = constants.TEST_TRANSFORMS(32)

    def get_model(self, arch):
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

class A2B(DataSet):
    def __init__(self, data_path, **kwargs):
        _, ds_name = os.path.split(data_path)
        valid_names = ['horse2zebra', 'apple2orange', 'summer2winter_yosemite']
        assert ds_name in valid_names, "path must end in one of {0}, not {1}".format(valid_names, ds_name)
        super(A2B, self).__init__(ds_name)
        self.data_path = data_path
        self.mean = constants.DEFAULT_MEAN
        self.std = constants.DEFAULT_STD
        self.num_classes = 2

        self.transform_train = constants.TRAIN_TRANSFORMS_224
        self.transform_test = constants.TEST_TRANSFORMS_224

    def get_model(self, arch):
        return models.__dict__[arch](num_classes=self.num_classes)

### Dictionary of datasets
DATASETS = {
    'imagenet': ImageNet,
    'restricted_imagenet': RestrictedImageNet,
    'restricted_imagenet_balanced': RestrictedImageNetBalanced,
    'cifar': CIFAR,
    'robust_cifar': ROBUST_CIFAR,
    'non_robust_cifar': NON_ROBUST_CIFAR,
    'cinic': CINIC,
    'a2b': A2B,
    'places': Places,
    'places_room': PlacesRoom,
    'places_filter_room': PlacesFilterRoom,
    'sun_lamp': SunLamp,
}
