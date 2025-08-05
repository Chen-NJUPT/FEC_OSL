__author__ = 'HPC'
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import warnings

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

import timm

# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import os
import PIL

from torchvision import datasets, transforms

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.utils.data import Subset

from engine import train_one_epoch_payload, evaluate
from data_process_payload import process_payload
import math

def build_dataset(args):
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    root = os.path.join(args.data_path)
    items = os.listdir(args.data_path)
    nb_classes = len(items)
    dataset = datasets.ImageFolder(root, transform=transform)
    total_size = len(dataset)
    nb_classes_train = int(nb_classes * args.ratio)
    train_unknow_classes = math.ceil(nb_classes_train / 3)
    know_classes = nb_classes_train - train_unknow_classes
    unknow_classes = nb_classes - know_classes
    index_know = (0, know_classes * int(len(dataset) / nb_classes))
    index_unknow_train = (know_classes * int(len(dataset) / nb_classes), 
                         know_classes * int(len(dataset) / nb_classes) + train_unknow_classes * int(len(dataset) / nb_classes))
    index_unknow = (know_classes * int(len(dataset) / nb_classes), total_size)
    
    # 当前只对已知类进行训练
    # 对数据集进行切片
    # 把最后的后几个当成全新未知类
    know_dataset = torch.utils.data.Subset(dataset, list(range(*index_know)))
    unknow_dataset_train = torch.utils.data.Subset(dataset, list(range(*index_unknow_train)))
    new_dataset = torch.utils.data.Subset(dataset, list(range(*index_unknow)))  # 未知类训练集，除了新类之外的未知类
    
    # 已知类划分
    train_size = int(0.8 * len(know_dataset))  # 第一轮 已知类 训练集大小  已知类训练
    val_size = len(know_dataset) - train_size  # 已知类 验证集大小
    train_dataset, val_dataset = random_split(know_dataset, [train_size, val_size])
    
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    
    # train未知类划分
    train_size2 = int(0.8 * len(unknow_dataset_train))
    val_size2 = len(unknow_dataset_train) - train_size2
    all_unknow_train_dataset, all_unknow_test_dataset  = random_split(unknow_dataset_train, [train_size2, val_size2])
    all_unknow_train_indices = all_unknow_train_dataset.indices
    all_unknow_test_indices = all_unknow_test_dataset.indices
    
    # test未知类划分
    train_size3 = int(0.8 * len(new_dataset))   # 弃
    val_size3 = len(new_dataset) - train_size3
    unknow_train_dataset, unknow_test_dataset  = random_split(new_dataset, [train_size3, val_size3])
    unknow_train_indices = unknow_train_dataset.indices
    unknow_test_indices = unknow_test_dataset.indices
    
    
    dic = {}
    dic["know_train_indices"] = train_indices
    dic["know_val_indices"] = val_indices

    dic["all_unknow_train_indices"] = [(i + len(know_dataset)) for i in all_unknow_train_indices]
    dic["all_unknow_val_indices"] = [(i + len(know_dataset)) for i in all_unknow_test_indices]

    offset = len(know_dataset) + len(unknow_dataset_train)

    unknow1_train = [(i + len(know_dataset)) for i in all_unknow_train_indices]
    unknow2_train = [(i + offset) for i in unknow_train_indices]
    combined_train = unknow1_train + unknow2_train
    np.random.shuffle(combined_train)

    unknow1_val = [(i + len(know_dataset)) for i in all_unknow_test_indices]
    unknow2_val = [(i + offset) for i in unknow_test_indices]
    combined_val = unknow1_val + unknow2_val
    np.random.shuffle(combined_val)

    dic["unknow_train_indices"] = combined_train
    dic["unknow_val_indices"] = combined_val
    
    with open('dataset/byte_data/data_index.json', 'w') as json_file:
        json.dump(dic, json_file, indent=4)  # indent=4 用于美化输出
    with open('dataset/byte_data/data_index.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        train_dataset = [dataset[i] for i in data["know_train_indices"]]
        val_dataset = [dataset[i] for i in data["know_val_indices"]]
        
        all_unknow_dataset_train = [dataset[i] for i in data["all_unknow_train_indices"]]
        all_unknow_dataset_val = [dataset[i] for i in data["all_unknow_val_indices"]]
        
        unknow_dataset_train = [dataset[i] for i in data["unknow_train_indices"]]
        unknow_dataset_val = [dataset[i] for i in data["unknow_val_indices"]]
        
    return train_dataset, val_dataset, all_unknow_dataset_train, unknow_dataset_val
