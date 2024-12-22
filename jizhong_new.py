import os
import argparse
import sys

from src import Range, set_logger, TensorBoardRunner, check_args, set_seed, load_dataset, load_model,get_best_gpu , load_dataset_m

import time
import copy
import numpy as np
import torch
import traceback
import torchvision.models
from tqdm import tqdm
from torchvision.transforms import transforms
from copy import deepcopy
from torch.utils.data import Subset
from torch import nn, optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--exp_name', help='name of the experiment', default='hx', type=str)
parser.add_argument('--seed', help='global random seed', type=int, default=5959)
parser.add_argument('--device', help='device to use; `cpu`, `cuda`, `cuda:GPU_NUMBER`', type=str, default='cuda')
parser.add_argument('--DATASET_PATH', help='path to save & read raw data', type=str, default='/home/hx/code/data')
parser.add_argument('--SAVE_PATH', help='path to save results', type=str, default='/home/hx/code/centralized/result')

parser.add_argument('--dataset', default='Cifar100', help='''name of dataset to use for an experiment (NOTE: case sensitive)
   - image classification datasets in `torchvision.datasets`,
   - among [ TinyImageNet | CINIC10  ] ''', type=str)
parser.add_argument('--test_size',help='a fraction of local hold-out dataset for evaluation (-1 for assigning pre-defined test split as local holdout set)',
                    type=float, choices=[Range(-1, 1.)], default=0.2)

parser.add_argument('--model_name', default='ResNet18', help='a model to be used (NOTE: case sensitive)', type=str,
                    choices=['LeNet','VGG9', 'VGG11', 'VGG13','ResNet10', 'ResNet18', 'ResNet34','alexnet','MobileNet','DenseNet-121'])
parser.add_argument('--dropout', help='dropout', type=float, default = 0.5 )
parser.add_argument('--hijack_dataset', help='hijack_dataset', type=str, default = 'Cifar10')
parser.add_argument('--pretrained', help='pretrained', type=int, default= 0 )
parser.add_argument('--save_model', help='save model', type=int, default= 0 )
parser.add_argument('--pretrained_path', help='pretrained_path', type=str,default=r'/home/hx/code/Federated-Learning-in-PyTorch-main/ckpt')
parser.add_argument('--noise_lr', help='noise_lr', type=float, default=0.01 )
parser.add_argument('--noised', help='noised', type=int, default = 1 )
parser.add_argument('--total_epochs', help='total_epochs', type=int, default=150)
parser.add_argument('--batch_size', help='batch_size', type=int, default=64)
parser.add_argument('--Lr', help='init——Lr', type=float, default = 0.1)
parser.add_argument('--hj_Lr', help='init——hj_Lr', type=float, default = 0.05)
parser.add_argument('--hijack', help='hijack', type=int, default = 1 )
parser.add_argument('--layered', help='layered', type=int, default = 1 )
parser.add_argument('--two_opt', help='two_opt', type=int, default = 0 )
parser.add_argument('--alpha', help='alpha', type=float, default = 0.1 )
parser.add_argument('--double_train', help='double_train', type=int, default = 1 )
parser.add_argument('--cifar', help='cifar', type=int, default = 0 )
parser.add_argument('--m', help='m', type=int, default = 20 )
parser.add_argument('--data_split', help='data_split', type=float, default = 1 )

args = parser.parse_args()
global_args = args

def main():

    print('*'*100,dir(torchvision.datasets))
    print('*'*100,torchvision.__file__)
    curr_time = time.strftime("%y%m%d_%H%M%S", time.localtime())

    args.SAVE_PATH = os.path.join(args.SAVE_PATH, f'{args.dataset}_{args.hijack_dataset}-model_name={args.model_name}-dropout={args.dropout}-noised={args.noised}'
                                                  f'-layered={args.layered}-batch_size={args.batch_size}'
                                                  f'-total_epochs={args.total_epochs}-noise_lr={args.noise_lr}'
                                                  f'-hj_lr={args.hj_Lr}-Lr={args.Lr}-pretrained={args.pretrained}-two_opt={args.two_opt}-alpha={args.alpha}-hijack={args.hijack}-double_train={args.double_train}-cifar={args.cifar}-m={args.m}-data_split={args.data_split}')
    filename = '{}{}-{}-noised={}-layered={}-batch_size={}-total_epochs={}-noise_lr={}-pretrained={}best_cnn_model-{}'.format(
        args.SAVE_PATH, args.dataset,
        args.hijack_dataset, args.noised, args.layered, args.batch_size,
        args.total_epochs, args.noise_lr, args.pretrained, curr_time)

    if not os.path.exists(args.SAVE_PATH):
        os.makedirs(args.SAVE_PATH)

    logger=set_logger(f'{args.SAVE_PATH}/{args.exp_name}_{curr_time}.log', args)
    logger.info(f'Arguments received:{args}')
    if args.cifar==1:
        if args.dataset == 'Cifar100':
            train_dataset,test_dataset = load_dataset_m(args,args.dataset,args.m)
            train_dataset_hj,test_dataset_hj=load_dataset(args,args.hijack_dataset)
        elif args.hijack_dataset == 'Cifar100':
            train_dataset, test_dataset = load_dataset(args, args.dataset)
            train_dataset_hj, test_dataset_hj = load_dataset_m(args, args.hijack_dataset,args.m)
    else:
        train_dataset,test_dataset = load_dataset(args,args.dataset)
        train_dataset_hj,test_dataset_hj=load_dataset(args,args.hijack_dataset)

    train_size = int(args.data_split * len(train_dataset_hj))
    test_size = len(train_dataset_hj) - train_size
    train_dataset_hj, _ = random_split(train_dataset_hj, [train_size, test_size])

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               )
    train_loader_hj = torch.utils.data.DataLoader(train_dataset_hj,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               )

    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             )
    val_loader_second = torch.utils.data.DataLoader(test_dataset_hj,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             )

    dc_dict={'Cifar10':10,'Cinic10':10,'Cifar100':100,'Mnist':10,'Imagenette':10,'DTD':47,'SVHN':10,'GTSRB':43}

    if args.model_name == "ResNet18":
        model_ft = torchvision.models.resnet18(pretrained=False, num_classes=dc_dict[args.dataset])
    elif args.model_name == "ResNet34":
        model_ft = torchvision.models.resnet34(pretrained=False, num_classes=dc_dict[args.dataset])
        model_ft.fc = nn.Sequential(nn.Dropout(args.dropout),nn.Linear(model_ft.fc.in_features, model_ft.fc.out_features))
    elif args.model_name == "VGG13":
        model_ft = torchvision.models.vgg13(pretrained=False, num_classes=dc_dict[args.dataset])
    elif args.model_name == "alexnet":
        model_ft = torchvision.models.alexnet(pretrained=False, num_classes=dc_dict[args.dataset])
    elif args.model_name == "MobileNet":
        model_ft = torchvision.models.mobilenet.MobileNetV2( num_classes=dc_dict[args.dataset])
    elif args.model_name == "DenseNet-121":
        model_ft = torchvision.models.densenet121(pretrained=False, num_classes=dc_dict[args.dataset])


    layer = torch.nn.Linear(dc_dict[args.dataset], dc_dict[args.hijack_dataset], bias=True)
    noise = init_noise(train_dataset_hj).to(device)
    if args.pretrained==1:

        model_ft_state_dict = torch.load(os.path.join(args.SAVE_PATH, f'best_model:{args.hijack_dataset}.pth'))
        model_ft.load_state_dict(model_ft_state_dict)

        layer_state_dict = torch.load(os.path.join(args.SAVE_PATH, f'best_layer:{args.hijack_dataset}.pth'))
        layer.load_state_dict(layer_state_dict)

        noise = torch.load(os.path.join(args.SAVE_PATH, f'best_noise:{args.hijack_dataset}.pth'))
        noise = noise.to(device)
        logger.info('Loaded model,layer,noise')

    model_ft.to(device)
    layer.to(device)

    loss_fn = nn.CrossEntropyLoss()

    counter = 0  
    counter_hj = 0
    since = time.time()
    best_acc = 0
    best_acc_second = 0
    epoch_hj = 0
    epoch_main = 0
    best_acc_third = 0
    optimizer = optim.SGD(model_ft.parameters(), lr=args.Lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(model_ft.parameters(), lr=args.Lr, weight_decay=5e-4)
    hijack_optimizer = optim.SGD(model_ft.parameters(), lr=args.hj_Lr, momentum=0.9, weight_decay=5e-4)
    layer_optimizer = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4 )
    noise_optimizer = optim.SGD([noise],lr=args.noise_lr,weight_decay=5e-4)

    logger.info('四轮不变减少一次学习率为之前的70%')
    for epoch in range(args.total_epochs):
        if counter / 4 == 1:
            counter = 0
            args.Lr = args.Lr * 0.7
        if counter_hj / 4 == 1:
            counter_hj = 0
            args.hj_Lr = args.hj_Lr * 0.7

        adjust_learning_rate(optimizer,args.Lr)

        print('-' * 10)
        logger.info(f'Epoch {epoch + 1}/{args.total_epochs}' )
        running_loss,running_corrects = train(train_loader=train_loader, model=model_ft, criterion=loss_fn,
                                              optimizer=optimizer,noised=0,layered=0,noise_optimizer=noise_optimizer,
                                              layer_optimizer=layer_optimizer,noise=noise,layer=layer)
        epoch_loss = running_loss / len(train_loader.sampler) 
        epoch_acc = float(running_corrects) / len(train_loader.sampler)
        time_elapsed = time.time() - since

        logger.info('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('{} Loss: {:.4f}[{}] Acc: {:.4f}'.format('train', epoch_loss, counter, epoch_acc))
        print()
        if args.hijack==1:
            if args.two_opt==1:
                adjust_learning_rate(hijack_optimizer, args.hj_Lr) 
            else:
                adjust_learning_rate(hijack_optimizer,args.Lr*args.alpha)
            running_loss, running_corrects = train(train_loader=train_loader_hj, model=model_ft, criterion=loss_fn,
                                                   optimizer=hijack_optimizer, noised=args.noised, layered=args.layered,
                                                   noise_optimizer=noise_optimizer, layer_optimizer=layer_optimizer,
                                                   noise=noise, layer=layer)

            epoch_loss_second = running_loss / len(train_loader_hj.sampler) 
            epoch_acc_second = float(running_corrects) / len(train_loader_hj.sampler)
            time_elapsed = time.time() - since

            logger.info('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            logger.info('{} Loss: {:.4f}[{}] Acc: {:.4f}'.format('train', epoch_loss_second, counter_hj, epoch_acc_second))
            print()
        if args.double_train ==1:
            adjust_learning_rate(optimizer, args.Lr*0.1)
            train(train_loader=train_loader, model=model_ft, criterion=loss_fn,
              optimizer=optimizer, noised=0, layered=0, noise_optimizer=noise_optimizer,
              layer_optimizer=layer_optimizer, noise=noise, layer=layer)
        # eval
        running_loss,running_corrects = eval(val_loader=val_loader,model=model_ft,criterion=loss_fn,noised=0,layered=0,noise=noise,layer=layer)

        epoch_loss_val = running_loss / len(val_loader.sampler)  
        epoch_acc_val = float(running_corrects) / len(val_loader.sampler)
        time_elapsed = time.time() - since
        logger.info('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('{} Loss: {:.4f}[{}] Acc: {:.4f}'.format('valid', epoch_loss_val, counter, epoch_acc_val))
        print()

        if args.hijack == 1:
            running_loss, running_corrects = eval(val_loader=val_loader_second, model=model_ft, criterion=loss_fn, noised=args.noised,
                                                  layered=args.layered, noise=noise, layer=layer)

            epoch_loss_val_second = running_loss / len(val_loader_second.sampler) 
            epoch_acc_val_second = float(running_corrects) / len(val_loader_second.sampler)
            time_elapsed = time.time() - since

            logger.info('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            logger.info('{} Loss_hijack: {:.4f}[{}] Acc_hijack: {:.4f}'.format('valid', epoch_loss_val_second, counter_hj, epoch_acc_val_second))
            print()


        if epoch_acc_val > best_acc:  # epoch_acc > best_acc:
            best_acc = epoch_acc_val
            epoch_main = epoch

            save_model(model_ft,args.dataset,args.SAVE_PATH)

            save_layer(layer,args.dataset,args.SAVE_PATH)

            save_noise(noise,args.dataset,args.SAVE_PATH)

            logger.info("已保存最优主任务模型，准确率:\033[1;31m {:.2f}%\033[0m(epoch:{})".format(best_acc * 100,epoch_main))
            logger.info("当前最高主任务准确率:{:.2f}%(epoch:{})，最高hijack任务准确率：{:.2f}(epcoh:{})".format(best_acc * 100,epoch_main,best_acc_second * 100 ,epoch_hj))
            counter = 0
        else:
            counter += 1
        if args.hijack==1:
            if epoch_acc_val_second > best_acc_second:
                best_acc_second= epoch_acc_val_second
                epoch_hj = epoch
                save_model(model_ft, args.hijack_dataset, args.SAVE_PATH)
                save_layer(layer, args.hijack_dataset, args.SAVE_PATH)
                save_noise(noise, args.hijack_dataset, args.SAVE_PATH)

                logger.info("已保存最优次要任务模型，准确率:\033[1;31m {:.2f}%\033[0m(epoch:{})".format(best_acc_second * 100,epoch_hj))
                logger.info("当前最高主任务准确率:{:.2f}%(epoch:{})，最高hijack任务准确率：{:.2f}%(epoch:{})".format(best_acc * 100,epoch_main,
                                                                                             best_acc_second * 100,epoch_hj))
                counter_hj = 0
            else:
                counter_hj += 1

        print()
        logger.info('当前主任务学习率 : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        logger.info('当前hijack任务学习率 : {:.7f}'.format(hijack_optimizer.param_groups[0]['lr']))
        print()

    time_elapsed = time.time() - since
    print()
    logger.info('任务完成！')
    logger.info('任务完成总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('最高主任务验证集准确率: {:4f}%(epoch:{})'.format(best_acc*100,epoch_main))
    logger.info('最高次任务验证集准确率: {:4f}%(epoch:{})'.format(best_acc_second*100,epoch_hj))
    # logger.info('最优情况下准确率: {:4f}%,{:4f}%'.format(best_acc_both*100,best_acc_second_both*100))

def init_noise(train_dataset):
    image, _ = train_dataset[-1]
    noise = torch.rand_like(image)
    if noise.shape[0] == 1:
        noise = noise.repeat(3, 1, 1)
    return noise

def save_model(model,dataset,path):
    best_model_wts = copy.deepcopy(model.state_dict())
    save_name_t = 'best_model:{}.pth'.format(dataset)
    save_file_path = os.path.join(path, save_name_t)
    torch.save(best_model_wts, save_file_path)

def save_layer(layer,dataset,path):
    best_layer_wts = copy.deepcopy(layer.state_dict())
    save_name_layer = 'best_layer:{}.pth'.format(dataset)
    save_layer_path = os.path.join(path, save_name_layer)
    torch.save(best_layer_wts, save_layer_path)

def save_noise(noise,dataset,path):
    best_noise = copy.deepcopy(noise)
    save_name_noise = 'best_noise:{}.pth'.format(dataset)
    save_noise_path = os.path.join(path, save_name_noise)
    torch.save(best_noise.cpu(), save_noise_path)

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, criterion, optimizer,layer_optimizer,noise_optimizer,noise,layer,noised,layered):
    running_loss=0
    running_corrects=0
    model.train()
    layer.train()
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        if noised == 1:
            inputs = inputs + noise
        optimizer.zero_grad()
        outputs = model(inputs)
        if layered == 1:
            layer_optimizer.zero_grad()
            outputs = layer(outputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        loss.backward() 
        if layered == 1:
            layer_optimizer.step()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)  
        running_corrects += (preds == labels).sum()  
    if noised == 1:
        running_loss = 0
        running_corrects = 0
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            if inputs.shape[1] == 1:
                inputs = inputs.repeat(1, 3, 1, 1)
            inputs = inputs + noise
            noise_optimizer.zero_grad()
            outputs = model(inputs)
            if layered == 1:
                outputs = layer(outputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward() 
            noise_optimizer.step()  
            running_loss += loss.item() * inputs.size(0)  
            running_corrects += (preds == labels).sum()  

    return running_loss,running_corrects

def eval(val_loader,model,criterion,noised,layered,noise,layer):
    running_loss = 0.0
    running_corrects = 0
    model.eval()
    layer.eval()
    with torch.no_grad():
        for val_images, val_labels in tqdm(val_loader):
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            if val_images.shape[1] == 1:
                val_images = val_images.repeat(1, 3, 1, 1)
            if noised == 1:
                val_images = val_images + noise

            outputs = model(val_images)
            if layered == 1:
                outputs = layer(outputs)
            loss = criterion(outputs, val_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            running_loss += loss.item() * val_images.size(0)
            running_corrects += (predict_y == val_labels).sum()  
    return running_loss,running_corrects

if __name__ == "__main__":

    device = args.device

    main()


