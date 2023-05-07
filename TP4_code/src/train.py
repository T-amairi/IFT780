#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License: Opensource, free to use
Other: Suggestions are welcome
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from manage.CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from manage.HDF5Dataset import HDF5Dataset
from models.AlexNet import AlexNet
from models.CNNVanilla import CnnVanilla
from models.ResNet import ResNet
from models.UNet import UNet
from models.yourUNet import yourUNet
from models.VggNet import VggNet
from models.yourSegNet import yourSegNet
from torchvision import datasets
from models.Loss import DiceLoss, TverskyLoss
from utils.utils import extract_data_augment


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 train.py --model UNet [hyper_parameters]'
                                           '\n python3 train.py --model UNet --predict',
                                     description="This program allows to train different models of classification on"
                                                 " different datasets. Be aware that when using UNet model there is no"
                                                 " need to provide a dataset since UNet model only train "
                                                 "on acdc dataset.",
                                     add_help=True)
    parser.add_argument('--model', type=str, default="UNet",
                        choices=["CnnVanilla", "VggNet", "AlexNet", "ResNet", "yourUNet", "yourSegNet", "UNet"])
    parser.add_argument('--dataset', type=str, default="acdc", choices=["cifar10", "svhn", "acdc"])
    parser.add_argument('--loss', type=str, default="CE", choices=["CE", "dice", "tversky"])
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--data_aug', type=int, default=0, choices=[0,1,2,3],
                        help="Data augmentation : 0 (not enabled), 1/2 (data aug mode type 1/2 for acdc), 3 (data aug for cifar10/svhn)")
    parser.add_argument('--predict', type=str,
                        help="Name of the file containing model weights used to make "
                             "segmentation prediction on test data")
    parser.add_argument('--plot', default=False, action='store_true',
                        help="Plot training metrics and/or 3 randoms prediction")
    parser.add_argument('--save', default=False, action='store_true', 
                        help="Save model weights after training")
    parser.add_argument('--enable_checkpoint', default=False, action='store_true', 
                        help="Enable checkpoint after each epoch")
    parser.add_argument('--load_checkpoint', default=False, action='store_true', 
                        help="Load a checkpoint")
    parser.add_argument('--n_extract_data_augment', type=int, default=0,
                        help="Make examples of the two mode of data augmentation. Number of pictures to extract")
    
    return parser.parse_args()


if __name__ == "__main__":

    args = argument_parser()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_set = args.validation
    learning_rate = args.lr
    data_augment = args.data_aug
    plot = args.plot
    save = args.save
    loss = args.loss
    enable_checkpoint = args.enable_checkpoint
    load_checkpoint = args.load_checkpoint
    n_extract_data_augment = args.n_extract_data_augment

    # handle exceptions
    if args.model == 'yourUNet' or args.model == 'yourSegNet' or args.model == 'UNet':
        if args.dataset != 'acdc':
            raise ValueError('Wrong dataset : you have to use the ACDC one')
    else:
        if args.dataset == 'acdc':
            raise ValueError('Wrong dataset : you have to use the CIFAR10 or SVHN one')
        
    if load_checkpoint and (not os.path.isfile('../weights/'+ args.model + '_checkpoint.pt') or not os.path.isfile('../weights/'+ args.model + '_checkpoint_weights.pt')):
        raise ValueError('There is no checkpoint to load')

    if data_augment:
        print('Data augmentation activated!')
    else:
        print('Data augmentation NOT activated!')

    if enable_checkpoint:
        print('Checkpointing is activated!')
    else:
        print('Checkpointing NOT activated!')

    # set hdf5 path according your hdf5 file location
    acdc_hdf5_file = '../data/ift780_acdc.hdf5'

    # Transform is used to normalize data among others
    acdc_base_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # random crop + norm
    acdc_data_augment_transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomHorizontalFlip(0.2),
        transforms.Normalize((0.5) ,(0.5))
    ])

    # 90 deg rotation + blur + norm
    acdc_data_augment_transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation((-20,20)),
        transforms.GaussianBlur(5),
        transforms.Normalize((0.5), (0.5))
    ])

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_augment_transform = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ])
   
    transform = base_transform
    acdc_transform = acdc_base_transform

    if n_extract_data_augment:
      if args.dataset == 'cifar10':
        test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=base_transform)
        test_set_no_norm = datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms.ToTensor())
        test_set_t1 = datasets.CIFAR10(root='../data', train=False, download=True, transform=acdc_data_augment_transform1)
        test_set_t2 = datasets.CIFAR10(root='../data', train=False, download=True, transform=acdc_data_augment_transform2)
      elif args.dataset == 'svhn':
        test_set = datasets.SVHN(root='../data', split='test', download=True, transform=base_transform)
        test_set_no_norm = datasets.SVHN(root='../data', split='test', download=True, transform=transforms.ToTensor())
        test_set_t1 = datasets.SVHN(root='../data', split='test', download=True, transform=acdc_data_augment_transform1)
        test_set_t2 = datasets.SVHN(root='../data', split='test', download=True, transform=acdc_data_augment_transform2)
      else:
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)
        test_set_no_norm = HDF5Dataset('test', acdc_hdf5_file, transform=transforms.ToTensor())
        test_set_t1 = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_data_augment_transform1)
        test_set_t2 = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_data_augment_transform2)
      
      extract_data_augment(test_set, test_set_no_norm, test_set_t1, test_set_t2, n_fig=n_extract_data_augment)
      quit()

    if data_augment == 1:
        acdc_transform = acdc_data_augment_transform1
    elif data_augment == 2:
        acdc_transform = acdc_data_augment_transform2
    elif data_augment == 3:
        transform = transforms.Compose([data_augment_transform, transform])

    if args.dataset == 'cifar10':
        # Download the train and test set and apply transform on it
        train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=base_transform)
    elif args.dataset == 'svhn':
        # Download the train and test set and apply transform on it
        train_set = datasets.SVHN(root='../data', split='train', download=True, transform=transform)
        test_set = datasets.SVHN(root='../data', split='test', download=True, transform=base_transform)
    else:
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=acdc_transform)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(torch.optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    if args.model == 'CnnVanilla':
        model = CnnVanilla(num_classes=10)
    elif args.model == 'AlexNet':
        model = AlexNet(num_classes=10)
    elif args.model == 'VggNet':
        model = VggNet(num_classes=10)
    elif args.model == 'ResNet':
        model = ResNet(num_classes=10)
    elif args.model == 'yourSegNet':
        model = yourSegNet(num_classes=4)
    elif args.model == 'yourUNet':
        model = yourUNet(num_classes=4)
    elif args.model == 'UNet':
        model = UNet(num_classes=4)

    if loss == "CE":
        loss_fn = nn.CrossEntropyLoss()
    elif loss == "dice":
        loss_fn = DiceLoss()
    else:
        loss_fn = TverskyLoss()

    model_trainer = CNNTrainTestManager(model=model,
                                        trainset=train_set,
                                        testset=test_set,
                                        batch_size=batch_size,
                                        loss_fn=loss_fn,
                                        optimizer_factory=optimizer_factory,
                                        validation=val_set,
                                        use_cuda=True,
                                        enable_checkpoint=enable_checkpoint,
                                        load_checkpoint=load_checkpoint)

    if args.predict is not None:
        print("predicting the mask of a randomly selected image from test set")
        model.load_weights(args.predict)
        if plot:
            model_trainer.plot_image_mask_prediction()
    else:
        print("Training {} on {} for {} epochs".format(model.__class__.__name__, args.dataset, args.num_epochs))
        model_trainer.train(num_epochs)
        model_trainer.evaluate_on_test_set()
        if save:
            model.save()
        if plot:
            model_trainer.plot_metrics()
