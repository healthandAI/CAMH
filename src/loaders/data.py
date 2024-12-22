import os
import gc
import torch
import logging
import random
from torchvision.transforms import Compose
import torchtext
import torchvision
import transformers
import concurrent.futures
from torch.utils.data import Subset
from torchvision.transforms import transforms
from src import TqdmToLogger, stratified_split
from src.datasets import *
from src.loaders.split import simulate_split

logger = logging.getLogger(__name__)

    

class SubsetWrapper(torch.utils.data.Dataset):
    """Wrapper of `torch.utils.data.Subset` module for applying individual transform.
    """
    def __init__(self, subset, suffix):
        self.subset = subset
        self.suffix = suffix

    def __getitem__(self, index):
        inputs, targets = self.subset[index]
        return inputs, targets

    def __len__(self):
        return len(self.subset)
    
    def __repr__(self):
        return f'{repr(self.subset.dataset.dataset)} {self.suffix}'

def load_dataset(args,dataset):
    def _get_transform():
        data_transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
                , torchvision.transforms.Resize((32,32),antialias=True)
                , torchvision.transforms.RandomCrop(32, padding=4)  
                , torchvision.transforms.RandomHorizontalFlip(p=0.5)  
                , torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
            ]),
            'valid': torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
                , torchvision.transforms.Resize((32,32),antialias=True)
                , torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        data_transforms_MNIST = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))  
            ]),
            'valid': torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
                , torchvision.transforms.Resize((32, 32), antialias=True)
                , torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]),
        }
        if dataset == 'Mnist':
            data_transforms = data_transforms_MNIST
        return data_transforms
    #
    # if args.dataset in torchvision.datasets.__dict__.keys():  # 3) for downloadable datasets in `torchvision.datasets`...
    #     data_transforms =_get_transform(args)
    #     train_dataset,val_dataset,_ = fetch_torchvision_dataset(args=args, dataset_name=args.dataset, root=args.DATASET_PATH, transforms=data_transforms)
    #################
    # fetch dataset #
    #################
    logger.info(f'[LOAD] Fetch dataset!')


    if dataset in "Cifar10": # 3) for downloadable datasets in `torchvision.datasets`...
        data_transforms = _get_transform()
        raw_train = torchvision.datasets.CIFAR10(root=args.DATASET_PATH, train=True,
                                                 download=True, transform=data_transforms["train"])
        raw_test = torchvision.datasets.CIFAR10(root=args.DATASET_PATH, train=False,
                                                 download=True, transform=data_transforms["valid"])

    elif dataset in "Cifar100": # 3) for downloadable datasets in `torchvision.datasets`...
        data_transforms = _get_transform()
        raw_train = torchvision.datasets.CIFAR100(root=args.DATASET_PATH, train=True,
                                                 download=True, transform=data_transforms["train"])
        raw_test = torchvision.datasets.CIFAR100(root=args.DATASET_PATH, train=False,
                                                download=True, transform=data_transforms["valid"])
    elif dataset in "Imagenette":  # 3) for downloadable datasets in `torchvision.datasets`...
        data_transforms = _get_transform()
        raw_train = torchvision.datasets.Imagenette(root=args.DATASET_PATH, split='train',
                                                  download=True, transform=data_transforms["train"])
        raw_test = torchvision.datasets.Imagenette(root=args.DATASET_PATH, split='val',
                                                 download=True, transform=data_transforms["valid"])
    elif dataset in "DTD":  # 3) for downloadable datasets in `torchvision.datasets`...
        data_transforms = _get_transform()
        raw_train = getattr(torchvision.datasets, 'DTD')(root=args.DATASET_PATH, split='train',
                                                    download=True, transform=data_transforms["train"])
        raw_test = getattr(torchvision.datasets, 'DTD')(root=args.DATASET_PATH, split='test',
                                                   download=True, transform=data_transforms["valid"])
    elif dataset in "SVHN":  # 3) for downloadable datasets in `torchvision.datasets`...
        data_transforms = _get_transform()
        raw_train = getattr(torchvision.datasets, 'SVHN')(root=args.DATASET_PATH, split='train',
                                                    download=True, transform=data_transforms["train"])
        raw_test = getattr(torchvision.datasets, 'SVHN')(root=args.DATASET_PATH, split='test',
                                                   download=True, transform=data_transforms["valid"])
    elif dataset in "GTSRB":  # 3) for downloadable datasets in `torchvision.datasets`...
        data_transforms = _get_transform()
        raw_train = getattr(torchvision.datasets, 'GTSRB')(root=args.DATASET_PATH, split='train',
                                                    download=True, transform=data_transforms["train"])
        raw_test = getattr(torchvision.datasets, 'GTSRB')(root=args.DATASET_PATH, split='test',
                                                   download=True, transform=data_transforms["valid"])
    elif dataset in "Mnist":  # 3) for downloadable datasets in `torchvision.datasets`...
        data_transforms = _get_transform()
        raw_train = getattr(torchvision.datasets, 'MNIST')(root=args.DATASET_PATH, train=True,
                                                    download=True, transform=data_transforms["train"])
        raw_test = getattr(torchvision.datasets, 'MNIST')(root=args.DATASET_PATH, train=False,
                                                   download=True, transform=data_transforms["valid"])

    # elif dataset == 'TinyImageNet': # 5) for other public datasets...
    #     transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
    #     raw_train, raw_test, args = fetch_tinyimagenet(args=args, root=args.data_path, transforms=transforms)
    #
    # elif dataset == 'CINIC10':
    #     transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
    #     raw_train, raw_test, args = fetch_cinic10(args=args, root=args.data_path, transforms=transforms)

    else: # x) for a dataset with no support yet or incorrectly entered...
        err = f'[LOAD] Dataset `{dataset}` is not supported or seems incorrectly entered... please check!'
        logger.exception(err)
        raise Exception(err)
    logger.info(f'[LOAD] ...successfully fetched dataset!')

    gc.collect()
    return  raw_train,raw_test


def load_dataset_m(args, dataset, m):
    """根据数据集名称加载一个新的数据集，其中只包含m个随机选择的类别"""
   
    raw_train,raw_test=load_dataset(args, dataset)
    classes = raw_train.targets
    # classes = classes.tolist()
    classes = list(set(classes))
    print('classes is', classes)
    # classes = list(set(raw_train.ta))
    # if m > len(classes):
    #     raise ValueError("m cannot be greater than the number of classes in the dataset")
    # print('classes is', classes)
    selected_classes = random.sample(classes, m)
    print('selected_classes is', selected_classes)


    subset_indices_train = [index for index, target in enumerate(raw_train.targets) if target in selected_classes]



    subset_indices_test = [index for index, target in enumerate(raw_test.targets) if target in selected_classes]


    subset_dataset_train = Subset(raw_train, subset_indices_train)
    subset_dataset_test = Subset(raw_test, subset_indices_test)
    print('subset_dataset_train is',len(subset_dataset_train))
    return subset_dataset_train, subset_dataset_test