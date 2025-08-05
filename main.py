from __future__ import division
__author__ = 'HPC'
import sys
import os
import argparse

base_dir = os.path.dirname(os.path.abspath(__file__))
byte_extract_path = os.path.join(base_dir, 'byte_features_extract')
sys.path.append(byte_extract_path)
flow_extract_path = os.path.join(base_dir, 'flow_features_extract')
sys.path.append(flow_extract_path)
energy_model_path = os.path.join(base_dir, 'energy_model')
sys.path.append(energy_model_path)

from byte_features_extract import header_dataset_generation, payload_dataset_generation, data_process_header, data_process_payload
from flow_features_extract import pacp_to_json

import torch.optim as optim
import os.path as osp
import time
from FEC_OSL_model import FECOSLModel
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

import numpy as np
import byte_features_extract.util.misc as misc
import byte_features_extract.util.lr_decay as lrd
from byte_features_extract.util.pos_embed import interpolate_pos_embed
from byte_features_extract.util.misc import NativeScalerWithGradNormCount as NativeScaler
from timm.data.mixup import Mixup
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import byte_features_extract.engine as engine
import byte_features_extract.header_dataset_generation as header_dataset_generation
import byte_features_extract.payload_dataset_generation as payload_dataset_generation
import datetime
from torch import nn
from tqdm import trange
import flow_features_extract.select_gpu as select_gpu
from torch.nn import functional as F
import h5py
import energy_model.energy as energy
from scipy.stats import weibull_min
import dgl
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import adjusted_mutual_info_score

def get_classes(path):
    items = os.listdir(path)
    for item in items:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            file_count = len([name for name in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, name))])
            print(item + ":" + str(file_count))
    nb_classes = len(items)
    return nb_classes

# byte_args
def get_args_parser_byte():
    parser = argparse.ArgumentParser('byte fine-tuning for traffic classification', add_help=False)
    # 64
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='FEC_OSL_byte_header_TransFormer', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=40, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    # parser.add_argument('--blr', type=float, default=2e-3, metavar='LR',
    #                     help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--blr', type=float, default=2e-6, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    # parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    #20
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    # parser.add_argument('--finetune', default='./output_dir/pretrained-model.pth',
    #                     help='finetune from checkpoint')
    parser.add_argument('--finetune', 
                        help='finetune from checkpoint')
    # parser.add_argument('--data_path', default='dataset/byte_data/tor_session/tor_session_PCAP_test', type=str,
    #                     help='dataset path')
    parser.add_argument('--data_path', default="dataset/byte_data/tor_session/tor_header", type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=8, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='dataset/output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

# header_model_train
def train_header_model(model, nb_classes, device,
        data_loader_byte_header_train_know, data_loader_byte_header_val_know,
        dataset_name, all_classes):
    args = get_args_parser_byte()
    args = args.parse_args()
    args.nb_classes = nb_classes
    args.epochs = 100
    args.blr = 2e-3
    args.device = device
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed) 
    cudnn.benchmark = True
    
    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
        
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model_without_ddp = model.byte_header_extractor
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()


    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    start_time = time.time()
    max_accuracy = 0.0
    max_f1 = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = engine.train_one_epoch(
            model_without_ddp, criterion, data_loader_byte_header_train_know,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        test_stats = engine.evaluate(data_loader_byte_header_val_know, model_without_ddp, device)
       
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        
        max_f1 = max(max_f1, test_stats["macro_f1"])
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    save_path = 'dataset/saved_models/' + dataset_name + '_byte_header_extractor_' + \
        str(nb_classes) + '_' + str(all_classes - nb_classes) + '.pth'
    torch.save(model.byte_header_extractor.state_dict(), save_path)
    print("成功保存" + "model.byte_header_extractor" + "模型到：" + 'dataset/saved_models/' + dataset_name + 
        '_byte_header_extractor_' + str(nb_classes) + '_' + str(all_classes - nb_classes) + '.pth')
    
# payload_model_train
def train_payload_model(model, nb_classes, device,
        data_loader_byte_payload_train_know, data_loader_byte_payload_val_know,
        dataset_name, all_classes):
    args = get_args_parser_byte()
    args = args.parse_args()
    args.nb_classes = nb_classes
    args.epochs = 100
    args.blr = 2e-3
    args.device = device
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed) 
    cudnn.benchmark = True
    
    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model_without_ddp = model.byte_payload_extractor
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    start_time = time.time()
    max_accuracy = 0.0
    max_f1 = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = engine.train_one_epoch(
            model_without_ddp, criterion, data_loader_byte_payload_train_know,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        test_stats = engine.evaluate(data_loader_byte_payload_val_know, model_without_ddp, device)
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        max_f1 = max(max_f1, test_stats["macro_f1"])
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    save_path = 'dataset/saved_models/' + dataset_name + '_byte_payload_extractor_' + \
        str(nb_classes) + '_' + str(all_classes - nb_classes) + '.pth'
    torch.save(model.byte_payload_extractor.state_dict(), save_path)

# flow_model_train
def train_flow_model(model, use_gpu, device, data_loader_flow_train_known, data_loader_flow_valid_known, 
    dataset_name, all_classes, nb_classes):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.flow_extractor.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    model.flow_extractor.train(True)
    epoch_losses = []
    epoch_acces = []
    epoch_precisions = []
    epoch_recalls = []
    epoch_f1s = []
    max_epoch = 100
    batch_size = 64
    for epoch in trange(max_epoch):
        epoch_loss = 0
        iter = 0
        while data_loader_flow_train_known.epoch_over == epoch:
            graphs, labels= data_loader_flow_train_known.next_train_batch_know(batch_size)  # 数据加载器获取下一个训练批次的图和标签。
            if use_gpu :
                graphs = graphs.to(torch.device(device))
                labels = labels.to(torch.device(device))
            predict_label = model.flow_extractor.forward_classify(graphs)
            loss = loss_func(predict_label, labels)
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            if use_gpu:
                lv= loss.detach().item()
            else:
                lv = loss.detach().cpu().item()
            epoch_loss += lv
            iter += 1
        epoch_loss /= (iter + 0.0000001)
        info='Epoch {}, loss: {:.4f}'.format(epoch, epoch_loss)
        epoch_losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        graphs, labels = data_loader_flow_valid_known.next_valid_batch_know(batch_size=batch_size)
        if use_gpu :
            graphs = graphs.to(torch.device(device))
            labels = labels.to(torch.device(device))
        predict_labels = model.flow_extractor.forward_classify(graphs)
        predict_labels = F.softmax(predict_labels,1)
        argmax_labels = torch.argmax(predict_labels,1)
        labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()  
        argmax_labels_np = argmax_labels.cpu().numpy() if argmax_labels.is_cuda else argmax_labels.numpy()
        precision = precision_score(labels_np, argmax_labels_np, average='macro') * 100  # 或'micro', 'weighted'  
        recall = recall_score(labels_np, argmax_labels_np, average='macro') * 100  # 或'micro', 'weighted'  
        f1 = f1_score(labels_np, argmax_labels_np, average='macro') * 100  # 或'micro', 'weighted'
        acc = (labels == argmax_labels).float().sum().item() / len(labels) * 100
        epoch_acces.append(acc)
        epoch_precisions.append(precision)
        epoch_recalls.append(recall)
        epoch_f1s.append(f1)
    save_path = 'dataset/saved_models/' + dataset_name + '_flow_extractor_' + \
        str(nb_classes) + '_' + str(all_classes - nb_classes) + '.pth'
    torch.save(model.flow_extractor.state_dict(), save_path)
    print("成功保存" + "model.flow_extractor" + "模型到：" + 'dataset/saved_models/' + dataset_name + 
        '_flow_extractor_' + str(nb_classes) + '_' + str(all_classes - nb_classes) + '.pth')

def adjust_loss_weights(epoch, max_epochs, start_weight=1.0, end_weight=0.1):
    return start_weight * (1 - epoch / max_epochs) + end_weight * (epoch / max_epochs)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

def Pre_train_ft(model, nb_classes, use_gpu, device, dataset_name, 
        data_loader_byte_header_train_know, data_loader_byte_header_val_know,
        data_loader_byte_payload_train_know, data_loader_byte_payload_val_know, 
        data_loader_flow_train_known, data_loader_flow_valid_known,
        data_loader_byte_header_train_unknow,
        data_loader_byte_payload_train_unknow,
        data_loader_flow_train_unknown,
        ratio):
    for param in model.byte_header_extractor.parameters():
        param.requires_grad = True
    for param in model.byte_payload_extractor.parameters():
        param.requires_grad = True
    for param in model.flow_extractor.parameters():
        param.requires_grad = True
    for param in model.energy_classifier.parameters():
        param.requires_grad = True
    for param in model.odc_classifier.parameters():
        param.requires_grad = True
    
    args = get_args_parser_byte()
    args = args.parse_args()
    args.nb_classes = int(nb_classes * ratio)
    args.epoch = 50
    seed = args.seed + misc.get_rank()  # 生成随机种子，使结果具有可复现性
    torch.manual_seed(seed)   # 设置PyTorch的随机种子
    np.random.seed(seed)      # 设置NumPy的随机种子
    cudnn.benchmark = True    # 使CuDNN在不同输入大小时找到最佳内核，提高计算效率
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
        
    model_without_ddp_header = model.byte_header_extractor
    model_without_ddp_payload = model.byte_payload_extractor
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
        
    param_groups_header = lrd.param_groups_lrd(model_without_ddp_header, args.weight_decay,
        no_weight_decay_list=model_without_ddp_header.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    
    param_groups_payload = lrd.param_groups_lrd(model_without_ddp_payload, args.weight_decay,
        no_weight_decay_list=model_without_ddp_payload.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer_byte_header = torch.optim.SGD(param_groups_header, lr=1e-4, momentum=0.9)
    optimizer_byte_payload = torch.optim.SGD(param_groups_payload, lr=1e-4, momentum=0.9)
    optimizer_flow = torch.optim.SGD(model.flow_extractor.parameters(), lr=1e-4, momentum=0.9)
    
    scheduler_flow = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_flow, mode='min', factor=0.5, patience=3, verbose=True)
    optimizer_energy = torch.optim.SGD(model.energy_classifier.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler_energy = torch.optim.lr_scheduler.LambdaLR(
        optimizer_energy,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epoch * len(data_loader_byte_header_train_know),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / 1e-3))
        
    scaler = GradScaler()  # 使用梯度标量进行混合精度训练
    
    loss_scaler = NativeScaler()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(args.epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    center_loss = CenterLoss(num_classes=nb_classes, feat_dim=256)
    Contrast_Loss = ContrastiveLoss()
    cross_entropy_loss = nn.CrossEntropyLoss()
    T = 10   # 能量损失的温度
    save_dataloader_to_h5(data_loader_byte_header_train_know, 'dataset/dataloader/' + 
        dataset_name + '_data_loader_byte_header_train_know_' + str(args.nb_classes) + "_" + str(nb_classes - args.nb_classes) + '.h5')
    save_dataloader_to_h5(data_loader_byte_header_train_unknow, 'dataset/dataloader/' + 
        dataset_name + '_data_loader_byte_header_train_unknow_' + str(args.nb_classes) + "_" + str(nb_classes - args.nb_classes) + '.h5')
    save_dataloader_to_h5(data_loader_byte_payload_train_know, 'dataset/dataloader/' + 
        dataset_name + '_data_loader_byte_payload_train_know_' + str(args.nb_classes) + "_" + str(nb_classes - args.nb_classes) + '.h5')
    save_dataloader_to_h5(data_loader_byte_payload_train_unknow, 'dataset/dataloader/' + 
        dataset_name + '_data_loader_byte_payload_train_unknow_' + str(args.nb_classes) + "_" + str(nb_classes - args.nb_classes) + '.h5')

    graphs_train_known = []
    labels_train_known = []
    graphs_train_unknown = []
    labels_train_unknown = []

    for i in data_loader_flow_train_known.know_train_index:
        graphs_train_known.append(data_loader_flow_train_known.graphs[i])
        labels_train_known.append(data_loader_flow_train_known.labelId[i])
    for i in data_loader_flow_train_unknown.unknow_train_index:
        graphs_train_unknown.append(data_loader_flow_train_unknown.graphs[i])
        labels_train_unknown.append(data_loader_flow_train_unknown.labelId[i])
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_train_know_' 
        + str(args.nb_classes) + "_" + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        feature_byte_header_train_know = torch.tensor(h5f['features'])
        label_byte_header_train_know = torch.tensor(h5f['labels'])
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_train_know_' 
        + str(args.nb_classes) + "_" + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        feature_byte_payload_train_know = torch.tensor(h5f['features'])
        label_byte_payload_train_know = torch.tensor(h5f['labels'])
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_train_unknow_' 
        + str(args.nb_classes) + "_" + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        feature_byte_header_train_unknow = torch.tensor(h5f['features'])
        label_byte_header_train_unknow = torch.tensor(h5f['labels'])
        label_byte_header_train_unknow[:] = int(nb_classes * ratio)
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_train_unknow_' 
        + str(args.nb_classes) + "_" + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        feature_byte_payload_train_unknow = torch.tensor(h5f['features'])
        label_byte_payload_train_unknow = torch.tensor(h5f['labels'])
        label_byte_payload_train_unknow[:] = int(nb_classes * ratio)

    feature_byte_header_train, label_byte_header_train = \
        merge_and_shuffle([feature_byte_header_train_know, feature_byte_header_train_unknow],
                        [label_byte_header_train_know, label_byte_header_train_unknow])
    feature_byte_payload_train, label_byte_payload_train = \
        merge_and_shuffle([feature_byte_payload_train_know, feature_byte_payload_train_unknow],
                        [label_byte_payload_train_know, label_byte_payload_train_unknow])
        
    unknow_len = len(labels_train_unknown)
    know_len = len(labels_train_known)
    labels_train_unknown = [int(nb_classes * ratio) for i in range(unknow_len)]
    flow_graphs_train, flow_labels_train = merge_and_shuffle_graphs(
        graphs_train_known + graphs_train_unknown,
        labels_train_known +  labels_train_unknown
    )
    for i in range(args.epoch):
        iterator_know_header = iter(data_loader_byte_header_train_know)
        iterator_know_payload = iter(data_loader_byte_payload_train_know)
        iterator_unknow_header = iter(data_loader_byte_header_train_unknow)
        iterator_unknow_payload = iter(data_loader_byte_payload_train_unknow)
        
        data_loader_flow_train_unknown.reflesh()  # 未知类也跟着重新刷新，因为为图片数据，不能像dataloader一样
        data_loader_flow_train_known.reflesh()
        
        optimizer_byte_header.zero_grad()
        optimizer_byte_payload.zero_grad()
        optimizer_flow.zero_grad()
        optimizer_energy.zero_grad()
        print("第" + str(i + 1) + "轮")
        print(f"forward: Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"forward: Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        batch_size = args.batch_size
        total_size = label_byte_header_train.shape[0]  # 记录训练总数
        batch_count = math.ceil(total_size / args.batch_size)
        pred_all = []
        target_all = []
        sigh_1 = 0
        sigh_2 = 0
        total_loss = 0  # 用于累积损失
        for j in range(0, total_size, batch_size):
            # 计算批次结束索引
            end = j + batch_size
            end_1 = sigh_1 + batch_size
            end_2 = sigh_2 + batch_size
            # 切片获取当前批次的特征和标签
            try:
                samples_header_train_know, targets_header_train_know = next(iterator_know_header)
                samples_payload_train_know, targets_payload_train_know = next(iterator_know_payload)
            except StopIteration:
                # 如果 loader_unknow 遍历完成，从头开始
                iterator_know_header = iter(data_loader_byte_header_train_know)
                iterator_know_payload = iter(data_loader_byte_payload_train_know)
                samples_header_train_know, targets_header_train_know = next(iterator_know_header)
                samples_payload_train_know, targets_payload_train_know = next(iterator_know_payload)
                
            try:
                samples_header_train_unknow, targets_header_train_unknow = next(iterator_unknow_header)
                samples_payload_train_unknow, targets_payload_train_unknow = next(iterator_unknow_payload)
                
            except StopIteration:
                # 如果 loader_unknow 遍历完成，从头开始
                iterator_unknow_header = iter(data_loader_byte_header_train_unknow)
                iterator_unknow_payload = iter(data_loader_byte_payload_train_unknow)
                samples_header_train_unknow, targets_header_train_unknow = next(iterator_unknow_header)
                samples_payload_train_unknow, targets_payload_train_unknow = next(iterator_unknow_payload)
            
            samples_header_train = feature_byte_header_train[j:end]
            targets_header_train = label_byte_header_train[j:end]
            samples_payload_train = feature_byte_payload_train[j:end]
            targets_payload_train = label_byte_header_train[j:end]
            
            graphs_train = dgl.batch(flow_graphs_train[j:end]).to(device)
            labels_train = flow_labels_train[j:end]
            if (sigh_1 % know_len) < (end_1 % know_len):
                graphs_train_know = dgl.batch(graphs_train_known[(sigh_1 % know_len):(end_1 % know_len)]).to(device)
                labels_train_know = torch.tensor(labels_train_known[(sigh_1 % know_len):(end_1 % know_len)])
                sigh_1 = end_1
            else:
                graphs_train_know = dgl.batch(graphs_train_known[(sigh_1 % know_len):know_len]).to(device)
                labels_train_know = torch.tensor(labels_train_known[(sigh_1 % know_len):know_len])
                sigh_1 = 0
            if (sigh_2 % unknow_len) < (end_2 % unknow_len):
                graphs_train_unknow = dgl.batch(graphs_train_unknown[(sigh_2 % unknow_len):(end_2 % unknow_len)]).to(device)
                labels_train_unknow = torch.tensor(labels_train_unknown[(sigh_2 % unknow_len):(end_2 % unknow_len)])
                sigh_2 = end_2
            else:
                graphs_train_unknow = dgl.batch(graphs_train_unknown[(sigh_2 % unknow_len):unknow_len]).to(device)
                labels_train_unknow = torch.tensor(labels_train_unknown[(sigh_2 % unknow_len):unknow_len])
                sigh_2 = 0
                
            if use_gpu :
                graphs_train = graphs_train.to(torch.device(device))
                labels_train = labels_train.to(torch.device(device))
                graphs_train_know = graphs_train_know.to(torch.device(device))
                labels_train_know = labels_train_know.to(torch.device(device))
                graphs_train_unknow = graphs_train_unknow.to(torch.device(device))
                labels_train_unknow = labels_train_unknow.to(torch.device(device))
                
            if mixup_fn is not None:
                samples_header_train, targets_header_train = mixup_fn(samples_header_train, targets_header_train)
                samples_payload_train, targets_payload_train = mixup_fn(samples_payload_train, targets_payload_train)
                samples_header_train_know, targets_header_train_know = mixup_fn(samples_header_train_know, targets_header_train_know)
                samples_payload_train_know, targets_payload_train_know = mixup_fn(samples_payload_train_know, targets_payload_train_know)
                samples_header_train_unknow, targets_header_train_unknow = mixup_fn(samples_header_train_unknow, targets_header_train_unknow)
                samples_payload_train_unknow, targets_payload_train_unknow = mixup_fn(samples_payload_train_unknow, targets_payload_train_unknow)
                
            with torch.cuda.amp.autocast(enabled=False):
                features, outputs = model(samples_header_train.to(device), samples_payload_train.to(device), graphs_train, 'energy')
                features_in, outputs_in = model(samples_header_train_know.to(device), samples_payload_train_know.to(device), graphs_train_know, 'energy')
                features_out, outputs_out = model(samples_header_train_unknow.to(device), samples_payload_train_unknow.to(device), graphs_train_unknow, 'energy')
                labels = labels_train

            loss = energy.energy_ft_loss(outputs, outputs_in, outputs_out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer_energy)
            
            scheduler_energy.step()
            scaler.step(optimizer_flow)
            scheduler_flow.step(loss)
            scaler.step(optimizer_byte_payload)
            scaler.step(optimizer_byte_header)
            scaler.update()
            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            _, y_train_pred = torch.max(probs, dim=1)
            pred_all.extend(y_train_pred.cpu().numpy().tolist())  # 转换为 NumPy 数组并追加
            target_all.extend(labels_train.cpu().numpy().tolist())  # 转换为 NumPy 数组并追加
            
        accuracy = accuracy_score(target_all, pred_all)
        precision = precision_score(target_all, pred_all, average='macro')
        recall = recall_score(target_all, pred_all, average='macro')
        f1 = f1_score(target_all, pred_all, average='macro')
        average_loss = total_loss / batch_count
        
        print(f"Epoch {i + 1} train Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Average Loss: {average_loss:.4f}")
        
        # 验证
        pred_all = []
        target_all = []
        total_loss = 0  # 用于累积损失
        data_loader_flow_valid_known.reflesh()
        for data_iter_step, ((samples_header_val, targets_header_val), 
                             (samples_payload_val  , targets_payload_val)) in enumerate(
                        zip(data_loader_byte_header_val_know, 
                            data_loader_byte_payload_val_know)):
            samples_header_val = samples_header_val.to(device, non_blocking=True)
            targets_header_val = targets_header_val.to(device, non_blocking=True)
            
            samples_payload_val = samples_payload_val.to(device, non_blocking=True)
            targets_payload_val = targets_payload_val.to(device, non_blocking=True)
            
            graphs_val, labels_val = data_loader_flow_valid_known.next_valid_batch_know(args.batch_size)  # 数据加载器获取下一个训练批次的图和标签。
            if use_gpu :
                graphs_val = graphs_val.to(torch.device(device))
                labels_val = labels_val.to(torch.device(device))
            if mixup_fn is not None:
                samples_header_val, targets_header_val = mixup_fn(samples_header_val, targets_header_val)
                samples_payload_val, targets_payload_val = mixup_fn(samples_payload_val, targets_payload_val)
            with torch.cuda.amp.autocast(enabled=False):
                _, outputs = model(samples_header_val, samples_payload_val, graphs_val, 'energy')
            
            loss = energy.energy_loss(outputs, labels_val, T)

            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)   # 将 logits 转换为概率分布
            _, y_val_pred = torch.max(probs, dim=1)  # 取最大概率对应的类别索引
            
            pred_all.extend(y_val_pred.cpu().numpy().tolist())  # 转换为 NumPy 数组并追加
            target_all.extend(labels_val.cpu().numpy().tolist())  # 转换为 NumPy 数组并追加
            
        accuracy = accuracy_score(target_all, pred_all)
        precision = precision_score(target_all, pred_all, average='macro')
        recall = recall_score(target_all, pred_all, average='macro')
        f1 = f1_score(target_all, pred_all, average='macro')
        average_loss = total_loss / batch_count
        print()
        print(f"Epoch {i + 1} valid Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Average Loss: {average_loss:.4f}")
        print("-" * 100)
        
    save_path = 'dataset/saved_models/' + dataset_name + '_Pre_FEC_OSL_model_' + \
        str(args.nb_classes) + "_" + str(nb_classes - args.nb_classes) + '.pth'
    torch.save(model, save_path)

    print("成功保存" + "model.flow_extractor" + "模型到：" + 'dataset/saved_models/' + 
        dataset_name + '_Pre_FEC_OSL_model_' + str(args.nb_classes) + "_" + str(nb_classes - args.nb_classes) + '.pth')

def train_ft(model, nb_classes, use_gpu, device, dataset_name, 
        data_loader_byte_header_train_know, data_loader_byte_header_val_know,
        data_loader_byte_payload_train_know, data_loader_byte_payload_val_know, 
        data_loader_flow_train_known, data_loader_flow_valid_known,
        data_loader_byte_header_train_unknow,
        data_loader_byte_payload_train_unknow,
        data_loader_flow_train_unknown,
        ratio):
    for param in model.byte_header_extractor.parameters():
        param.requires_grad = True
    for param in model.byte_payload_extractor.parameters():
        param.requires_grad = True
    for param in model.flow_extractor.parameters():
        param.requires_grad = True
    for param in model.energy_classifier.parameters():
        param.requires_grad = True
    for param in model.odc_classifier.parameters():
        param.requires_grad = True
    
    args = get_args_parser_byte()
    args = args.parse_args()
    args.nb_classes = int(nb_classes * ratio)
    args.epoch = 10
    seed = args.seed + misc.get_rank()  # 生成随机种子，使结果具有可复现性
    torch.manual_seed(seed)   # 设置PyTorch的随机种子
    np.random.seed(seed)      # 设置NumPy的随机种子
    cudnn.benchmark = True    # 使CuDNN在不同输入大小时找到最佳内核，提高计算效率
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
        
    model_without_ddp_header = model.byte_header_extractor
    model_without_ddp_payload = model.byte_payload_extractor
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
        
    param_groups_header = lrd.param_groups_lrd(model_without_ddp_header, args.weight_decay,
        no_weight_decay_list=model_without_ddp_header.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    
    param_groups_payload = lrd.param_groups_lrd(model_without_ddp_payload, args.weight_decay,
        no_weight_decay_list=model_without_ddp_payload.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer_byte_header = torch.optim.SGD(param_groups_header, lr=1e-4, momentum=0.9)
    optimizer_byte_payload = torch.optim.SGD(param_groups_payload, lr=1e-4, momentum=0.9)
    optimizer_flow = torch.optim.SGD(model.flow_extractor.parameters(), lr=1e-4, momentum=0.9)
    
    scheduler_flow = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_flow, mode='min', factor=0.5, patience=3, verbose=True)
    optimizer_energy = torch.optim.SGD(model.energy_classifier.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler_energy = torch.optim.lr_scheduler.LambdaLR(
        optimizer_energy,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epoch * len(data_loader_byte_header_train_know),
            1,
            1e-6 / 1e-3))
    
    optimizer_odc_classifier = torch.optim.SGD(model.odc_classifier.parameters(), lr=0.15, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler_odc_classifier = torch.optim.lr_scheduler.LambdaLR(
        optimizer_odc_classifier,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epoch * len(data_loader_byte_header_train_unknow),
            1,
            1e-6 / 1e-3))
    
    scaler = GradScaler()
    
    loss_scaler = NativeScaler()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(args.epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    center_loss = CenterLoss(num_classes=nb_classes, feat_dim=256)
    Contrast_Loss = ContrastiveLoss()
    cross_entropy_loss = nn.CrossEntropyLoss()
    T = 10   # 能量损失的温度
    save_dataloader_to_h5(data_loader_byte_header_train_know, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_header_train_know_' + str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5')
    save_dataloader_to_h5(data_loader_byte_header_train_unknow, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_header_train_unknow_' + str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5')
    
    save_dataloader_to_h5(data_loader_byte_payload_train_know, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_payload_train_know_' + str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5')
    save_dataloader_to_h5(data_loader_byte_payload_train_unknow, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_payload_train_unknow_' + str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5')

    graphs_train_known = []
    labels_train_known = []
    graphs_train_unknown = []
    labels_train_unknown = []

    for i in data_loader_flow_train_known.know_train_index:
        graphs_train_known.append(data_loader_flow_train_known.graphs[i])
        labels_train_known.append(data_loader_flow_train_known.labelId[i])
    for i in data_loader_flow_train_unknown.unknow_train_index:
        graphs_train_unknown.append(data_loader_flow_train_unknown.graphs[i])
        labels_train_unknown.append(data_loader_flow_train_unknown.labelId[i])
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_train_know_' + 
        str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        feature_byte_header_train_know = torch.tensor(h5f['features'])
        label_byte_header_train_know = torch.tensor(h5f['labels'])
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_train_know_' + 
        str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        feature_byte_payload_train_know = torch.tensor(h5f['features'])
        label_byte_payload_train_know = torch.tensor(h5f['labels'])
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_train_unknow_' + 
        str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        feature_byte_header_train_unknow = torch.tensor(h5f['features'])
        label_byte_header_train_unknow = torch.tensor(h5f['labels'])  # 未知类当成一个类
        label_byte_header_train_unknow[:] = int(nb_classes * ratio)
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_train_unknow_' + 
        str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        feature_byte_payload_train_unknow = torch.tensor(h5f['features']) # 未知类当成一个类
        label_byte_payload_train_unknow = torch.tensor(h5f['labels'])
        label_byte_payload_train_unknow[:] = int(nb_classes * ratio)

    # 未知类真实标签
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_train_unknow_' + 
        str(args.nb_classes) + '_' + str(nb_classes - args.nb_classes) + '.h5', 'r') as h5f:
        true_unknow_labels = torch.tensor(h5f['labels'])
        
    with torch.no_grad():
        
        features_all, outputs_all = model(feature_byte_header_train_unknow.to(device), feature_byte_payload_train_unknow.to(device), dgl.batch(graphs_train_unknown).to(device), 'odc')
        
        pred_labels = model.odc_classifier.initialize_memory_bank(features_all, num_clusters=nb_classes - int(nb_classes * ratio)) # 未知类别数目  (1 - ratio)
        pred_labels_numpy = pred_labels.cpu().numpy()
        true_unknow_labels_numpy = true_unknow_labels.cpu().numpy()
        ami_score = adjusted_mutual_info_score(true_unknow_labels_numpy, pred_labels_numpy)
        print(f"Initial Kmeans ami-score:{ami_score}")
        
    feature_byte_header_train, label_byte_header_train = \
        merge_and_shuffle([feature_byte_header_train_know, feature_byte_header_train_unknow],
                        [label_byte_header_train_know, label_byte_header_train_unknow])
    feature_byte_payload_train, label_byte_payload_train = \
        merge_and_shuffle([feature_byte_payload_train_know, feature_byte_payload_train_unknow],
                        [label_byte_payload_train_know, label_byte_payload_train_unknow])
        
    unknow_len = len(labels_train_unknown)
    know_len = len(labels_train_known)
    labels_train_unknown = [int(nb_classes * ratio) for i in range(unknow_len)]  # 未知类当成一个类
    flow_graphs_train, flow_labels_train = merge_and_shuffle_graphs(
        graphs_train_known + graphs_train_unknown,
        labels_train_known +  labels_train_unknown
    )
    data_list_scaled = []
    target_list = []
    title_list = []
    for i in range(args.epoch):
        iterator_know_header = iter(data_loader_byte_header_train_know)
        iterator_know_payload = iter(data_loader_byte_payload_train_know)
        iterator_unknow_header = iter(data_loader_byte_header_train_unknow)
        iterator_unknow_payload = iter(data_loader_byte_payload_train_unknow)
        
        data_loader_flow_train_unknown.reflesh()  # 未知类也跟着重新刷新，因为为图片数据，不能像dataloader一样
        data_loader_flow_train_known.reflesh()
        
        optimizer_byte_header.zero_grad()
        optimizer_byte_payload.zero_grad()
        optimizer_flow.zero_grad()
        optimizer_energy.zero_grad()
        print("第" + str(i + 1) + "轮")
        print(f"forward: Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"forward: Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        batch_size = args.batch_size
        total_size = label_byte_header_train.shape[0]  # 记录训练总数
        batch_count = math.ceil(total_size / args.batch_size)
        pred_all = []
        target_all = []
        sigh_1 = 0  # known
        sigh_2 = 0  # energy
        sigh_3 = 0  # ODC
        total_loss = 0
        cluster_labels = []
        true_unknown_labels_list = []
        features_list = []
        for j in range(0, total_size, batch_size):
            end = j + batch_size
            end_1 = sigh_1 + batch_size
            end_2 = sigh_2 + batch_size
            end_3 = sigh_3 + batch_size
            try:
                samples_header_train_know, targets_header_train_know = next(iterator_know_header)
                samples_payload_train_know, targets_payload_train_know = next(iterator_know_payload)
            except StopIteration:
                iterator_know_header = iter(data_loader_byte_header_train_know)
                iterator_know_payload = iter(data_loader_byte_payload_train_know)
                samples_header_train_know, targets_header_train_know = next(iterator_know_header)
                samples_payload_train_know, targets_payload_train_know = next(iterator_know_payload)
                
            try:
                samples_header_train_unknow, targets_header_train_unknow = next(iterator_unknow_header)
                samples_payload_train_unknow, targets_payload_train_unknow = next(iterator_unknow_payload)
                
            except StopIteration:
                iterator_unknow_header = iter(data_loader_byte_header_train_unknow)
                iterator_unknow_payload = iter(data_loader_byte_payload_train_unknow)
                samples_header_train_unknow, targets_header_train_unknow = next(iterator_unknow_header)
                samples_payload_train_unknow, targets_payload_train_unknow = next(iterator_unknow_payload)
            
            samples_header_train = feature_byte_header_train[j:end]
            targets_header_train = label_byte_header_train[j:end]
            samples_payload_train = feature_byte_payload_train[j:end]
            targets_payload_train = label_byte_header_train[j:end]
            
            graphs_train = dgl.batch(flow_graphs_train[j:end]).to(device)
            labels_train = flow_labels_train[j:end]
            if (sigh_1 % know_len) < (end_1 % know_len):
                graphs_train_know = dgl.batch(graphs_train_known[(sigh_1 % know_len):(end_1 % know_len)]).to(device)
                labels_train_know = torch.tensor(labels_train_known[(sigh_1 % know_len):(end_1 % know_len)])
                sigh_1 = end_1
            else:
                # 保持统一，从头开始
                graphs_train_know = dgl.batch(graphs_train_known[(sigh_1 % know_len):know_len]).to(device)
                labels_train_know = torch.tensor(labels_train_known[(sigh_1 % know_len):know_len])
                sigh_1 = 0
            if (sigh_2 % unknow_len) < (end_2 % unknow_len):
                graphs_train_unknow = dgl.batch(graphs_train_unknown[(sigh_2 % unknow_len):(end_2 % unknow_len)]).to(device)
                labels_train_unknow = torch.tensor(labels_train_unknown[(sigh_2 % unknow_len):(end_2 % unknow_len)])
                sigh_2 = end_2
            else:
                graphs_train_unknow = dgl.batch(graphs_train_unknown[(sigh_2 % unknow_len):unknow_len]).to(device)
                labels_train_unknow = torch.tensor(labels_train_unknown[(sigh_2 % unknow_len):unknow_len])
                sigh_2 = 0
                
            if use_gpu :
                graphs_train = graphs_train.to(torch.device(device))
                labels_train = labels_train.to(torch.device(device))
                graphs_train_know = graphs_train_know.to(torch.device(device))
                labels_train_know = labels_train_know.to(torch.device(device))
                graphs_train_unknow = graphs_train_unknow.to(torch.device(device))
                labels_train_unknow = labels_train_unknow.to(torch.device(device))
                
            if mixup_fn is not None:
                samples_header_train, targets_header_train = mixup_fn(samples_header_train, targets_header_train)
                samples_payload_train, targets_payload_train = mixup_fn(samples_payload_train, targets_payload_train)
                samples_header_train_know, targets_header_train_know = mixup_fn(samples_header_train_know, targets_header_train_know)
                samples_payload_train_know, targets_payload_train_know = mixup_fn(samples_payload_train_know, targets_payload_train_know)
                samples_header_train_unknow, targets_header_train_unknow = mixup_fn(samples_header_train_unknow, targets_header_train_unknow)
                samples_payload_train_unknow, targets_payload_train_unknow = mixup_fn(samples_payload_train_unknow, targets_payload_train_unknow)
                
            with torch.cuda.amp.autocast(enabled=False):
                features_all, outputs = model(samples_header_train.to(device), samples_payload_train.to(device), graphs_train, 'energy')
                features_in, outputs_in = model(samples_header_train_know.to(device), samples_payload_train_know.to(device), graphs_train_know, 'energy')
                features, outputs_out = model(samples_header_train_unknow.to(device), samples_payload_train_unknow.to(device), graphs_train_unknow, 'energy')
                labels = labels_train
            loss = energy.energy_ft_loss(outputs, outputs_in, outputs_out, labels) # new
            
            if (sigh_3 % unknow_len) < (end_3 % unknow_len):
                idx = torch.arange((sigh_3 % unknow_len), (end_3 % unknow_len))
                true_unknown_labels_list.append(true_unknow_labels[(sigh_3 % unknow_len):(end_3 % unknow_len)])
                sigh_3 = end_3
            else:
                idx = torch.arange((sigh_3 % unknow_len), unknow_len)
                true_unknown_labels_list.append(true_unknow_labels[(sigh_3 % unknow_len):unknow_len])
                sigh_3 = 0
                
            cls_score = model.odc_classifier(features)
            
            pseudo_labels = model.odc_classifier.memory_bank['labels'][idx].clone().detach().to(dtype=torch.long).to(cls_score[0].device)
            losses = model.odc_classifier.loss(features, model.odc_classifier.memory_bank['centroids'], cls_score, pseudo_labels) 
            change_ratio = model.odc_classifier.update_memory_bank_samples(idx, features.detach())
            losses['change_ratio'] = change_ratio
            loss_ODC = model.odc_classifier.parse_losses(losses)
            loss += loss_ODC

            scaler.scale(loss).backward()
            scaler.step(optimizer_odc_classifier)
            scheduler_odc_classifier.step()
            scaler.step(optimizer_energy)
            scheduler_energy.step()
            scaler.step(optimizer_flow) 
            scheduler_flow.step(loss)
            scaler.step(optimizer_byte_payload)
            scaler.step(optimizer_byte_header)
            scaler.update()
            total_loss += loss.item()
            
            
            max_values, predicted_labels = torch.max(cls_score[0], 1)

            features_list.append(features.detach())
            cluster_labels.append(predicted_labels)
            model.odc_classifier.update_centroids()
            model.odc_classifier.deal_with_small_clusters()
            
            _, y_train_pred = torch.max(outputs, dim=1)
            pred_all.extend(y_train_pred.cpu().numpy().tolist())
            target_all.extend(labels_train.cpu().numpy().tolist())
        
        accuracy = accuracy_score(target_all, pred_all)
        precision = precision_score(target_all, pred_all, average='macro')
        recall = recall_score(target_all, pred_all, average='macro')
        f1 = f1_score(target_all, pred_all, average='macro')
        average_loss = total_loss / batch_count

        print(f"Epoch {i + 1} train Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Average Loss: {average_loss:.4f}")

        
        # 验证
        pred_all = []
        target_all = []
        total_loss = 0  # 用于累积损失
        data_loader_flow_valid_known.reflesh()
        for data_iter_step, ((samples_header_val, targets_header_val), 
                             (samples_payload_val  , targets_payload_val)) in enumerate(
                        zip(data_loader_byte_header_val_know, 
                            data_loader_byte_payload_val_know)):
            samples_header_val = samples_header_val.to(device, non_blocking=True)
            targets_header_val = targets_header_val.to(device, non_blocking=True)
            
            samples_payload_val = samples_payload_val.to(device, non_blocking=True)
            targets_payload_val = targets_payload_val.to(device, non_blocking=True)
            
            graphs_val, labels_val = data_loader_flow_valid_known.next_valid_batch_know(args.batch_size)  # 数据加载器获取下一个训练批次的图和标签。
            if use_gpu :
                graphs_val = graphs_val.to(torch.device(device))
                labels_val = labels_val.to(torch.device(device))
            if mixup_fn is not None:
                samples_header_val, targets_header_val = mixup_fn(samples_header_val, targets_header_val)
                samples_payload_val, targets_payload_val = mixup_fn(samples_payload_val, targets_payload_val)
            with torch.cuda.amp.autocast(enabled=False):
                _, outputs = model(samples_header_val, samples_payload_val, graphs_val, 'energy')
            
            loss = energy.energy_loss(outputs, labels_val, T) 

            total_loss += loss.item()
            _, y_val_pred = torch.max(outputs, dim=1)
            pred_all.extend(y_val_pred.cpu().numpy().tolist())  # 转换为 NumPy 数组并追加
            target_all.extend(labels_val.cpu().numpy().tolist())  # 转换为 NumPy 数组并追加
            
        accuracy = accuracy_score(target_all, pred_all)
        precision = precision_score(target_all, pred_all, average='macro')
        recall = recall_score(target_all, pred_all, average='macro')
        f1 = f1_score(target_all, pred_all, average='macro')
        average_loss = total_loss / batch_count
        print()
        print(f"Epoch {i + 1} valid Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Average Loss: {average_loss:.4f}")
        print("-" * 100)

        
        new_labels = model.odc_classifier.memory_bank['labels']
        if new_labels.is_cuda:
            new_labels = new_labels.cpu()
        histogram = np.bincount(new_labels, minlength=nb_classes - int(nb_classes*ratio))  # nb_classes*(1 - ratio)
        non_empty_clusters = np.count_nonzero(histogram)
        print(f"Number of non-empty clusters: {non_empty_clusters}")
        print(f"Cluster sizes: {histogram}")
        
        cluster_labels_tensor = torch.cat(cluster_labels, dim=0)
        # 转换为NumPy数组
        cluster_labels_numpy = cluster_labels_tensor.cpu().numpy()
        true_unknow_labels_tensor = torch.cat(true_unknown_labels_list, dim=0)
        true_unknow_labels_numpy = true_unknow_labels_tensor.cpu().numpy()

        ami_score = adjusted_mutual_info_score(true_unknow_labels_numpy, cluster_labels_numpy)
        print(f'Test ami_score: {ami_score:.2f}')  # 输出测试结果

# dataset
def get_byte_header_dataset(dataset_header_path, ratio, new_classes):
    args = get_args_parser_byte()
    args = args.parse_args()
    args.data_path = dataset_header_path
    args.ratio = ratio
    args.new_classes = new_classes
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    dataset_train, dataset_val, unknow_dataset_train, unknow_dataset_val = header_dataset_generation.build_dataset(args=args)
    sampler_know_train = torch.utils.data.SequentialSampler(dataset_train)  # 修改为顺序执行
    sampler_know_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_unknow_train = torch.utils.data.SequentialSampler(unknow_dataset_train)
    sampler_unknow_val = torch.utils.data.SequentialSampler(unknow_dataset_val)
    data_loader_train_know = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_know_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val_know = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_know_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    data_loader_train_unknow = torch.utils.data.DataLoader(
        unknow_dataset_train, sampler=sampler_unknow_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    data_loader_val_unknow = torch.utils.data.DataLoader(
        unknow_dataset_val, sampler=sampler_unknow_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return data_loader_train_know, data_loader_val_know, data_loader_train_unknow, data_loader_val_unknow

def get_byte_payload_dataset(dataset_payload_path, ratio, new_classes):
    args = get_args_parser_byte()
    args = args.parse_args()
    args.data_path = dataset_payload_path
    device = torch.device(args.device)
    args.ratio = ratio
    args.new_classes = new_classes
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    dataset_train, dataset_val, unknow_dataset_train, unknow_dataset_val = payload_dataset_generation.build_dataset(args=args)
    sampler_know_train = torch.utils.data.SequentialSampler(dataset_train)
    sampler_know_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_unknow_train = torch.utils.data.SequentialSampler(unknow_dataset_train)
    sampler_unknow_val = torch.utils.data.SequentialSampler(unknow_dataset_val)
    data_loader_train_know = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_know_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val_know = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_know_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    data_loader_train_unknow = torch.utils.data.DataLoader(
        unknow_dataset_train, sampler=sampler_unknow_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    data_loader_val_unknow = torch.utils.data.DataLoader(
        unknow_dataset_val, sampler=sampler_unknow_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,    # 保留尾部数据
    )
    return data_loader_train_know, data_loader_val_know, data_loader_train_unknow, data_loader_val_unknow
    
def get_flow_dataset(dataset_flow_path, all_classes, nb_classes):
    data_loader_flow_train_known = flow_main_model.model(dataset_flow_path, randseed=256, splitrate=0.1,
        all_classes = all_classes, nb_classes = nb_classes).parse_raw_data(all_classes, nb_classes)
    data_loader_flow_valid_known = flow_main_model.model(dataset_flow_path, randseed=256, splitrate=0.1,
        all_classes = all_classes, nb_classes = nb_classes).parse_raw_data(all_classes, nb_classes)
    data_loader_flow_train_unknown = flow_main_model.model(dataset_flow_path, randseed=256, splitrate=0.1,
        all_classes = all_classes, nb_classes = nb_classes).parse_raw_data(all_classes, nb_classes)
    data_loader_flow_valid_unknown = flow_main_model.model(dataset_flow_path, randseed=256, splitrate=0.1,
        all_classes = all_classes, nb_classes = nb_classes).parse_raw_data(all_classes, nb_classes)
    return data_loader_flow_train_known, data_loader_flow_valid_known, \
        data_loader_flow_train_unknown, data_loader_flow_valid_unknown

def save_dataloader_to_h5(data_loader, h5_filename):
    if os.path.exists(h5_filename):
        print(f"Data  {h5_filename} 已存在") 
        return
    # 打开一个 HDF5 文件用于写入数据
    with h5py.File(h5_filename, 'w') as h5f:
        # 计算数据集总大小
        dataset_size = len(data_loader.dataset)
        batch_size = data_loader.batch_size

        # 获取第一批数据，用于推断数据形状
        first_batch = next(iter(data_loader))
        features_shape = first_batch[0].shape[1:]  # 忽略 batch_size 维度，只需要特征的形状
        labels_shape = first_batch[1].shape[1:]  # 忽略 batch_size 维度，只需要标签的形状

        # 创建用于存储特征和标签的数据集
        features_dataset = h5f.create_dataset('features', shape=(dataset_size,) + features_shape, dtype='float32')
        labels_dataset = h5f.create_dataset('labels', shape=(dataset_size,) + labels_shape, dtype='int64')

        # 遍历 DataLoader，将每个 batch 的数据存储到 HDF5 文件中
        idx = 0  # 用于跟踪当前存储的位置
        for batch_idx, (features, labels) in enumerate(data_loader):
            # 获取当前 batch 的大小（最后一个 batch 可能小于指定的 batch_size）
            current_batch_size = features.size(0)
            
            # 将数据移动到 CPU 并转换为 NumPy 数组以进行存储
            features_np = features.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            # 将数据存储到相应的数据集位置中
            features_dataset[idx:idx + current_batch_size] = features_np
            labels_dataset[idx:idx + current_batch_size] = labels_np

            # 更新索引位置
            idx += current_batch_size

        print(f"Data successfully saved to {h5_filename}")

# energy_iteration
def train_iteration_energy_ft(model, nb_classes, use_gpu, device, 
        data_loader_byte_header_train_know, data_loader_byte_header_val_know,
        data_loader_byte_payload_train_know, data_loader_byte_payload_val_know, 
        data_loader_flow_train_known, data_loader_flow_valid_known,
        data_loader_byte_header_train_unknow,
        data_loader_byte_payload_train_unknow,
        data_loader_flow_train_unknown, iter_epoch, ratio, MAVs, thresholds):
    
    for param in model.byte_header_extractor.parameters():
        param.requires_grad = True
    for param in model.byte_payload_extractor.parameters():
        param.requires_grad = True
    for param in model.flow_extractor.parameters():
        param.requires_grad = True
    for param in model.energy_classifier.parameters():
        param.requires_grad = True
    for param in model.odc_classifier.parameters():
        param.requires_grad = False
    
    args = get_args_parser_byte()
    args = args.parse_args()
    args.nb_classes = int(nb_classes * ratio)
    args.epoch = 10
    seed = args.seed + misc.get_rank()  # 生成随机种子，使结果具有可复现性
    torch.manual_seed(seed)   # 设置PyTorch的随机种子
    np.random.seed(seed)      # 设置NumPy的随机种子
    cudnn.benchmark = True    # 使CuDNN在不同输入大小时找到最佳内核，提高计算效率
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
        
    model_without_ddp_header = model.byte_header_extractor
    model_without_ddp_payload = model.byte_payload_extractor
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
        
    param_groups_header = lrd.param_groups_lrd(model_without_ddp_header, args.weight_decay,
        no_weight_decay_list=model_without_ddp_header.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    # optimizer_byte_header = torch.optim.AdamW(param_groups_header, lr=args.lr)
    
    param_groups_payload = lrd.param_groups_lrd(model_without_ddp_payload, args.weight_decay,
        no_weight_decay_list=model_without_ddp_payload.no_weight_decay(),
        layer_decay=args.layer_decay
    )

    optimizer_byte_header = torch.optim.SGD(param_groups_header, lr=1e-4, momentum=0.9)

    optimizer_byte_payload = torch.optim.SGD(param_groups_payload, lr=1e-4, momentum=0.9)

    optimizer_flow = torch.optim.SGD(model.flow_extractor.parameters(), lr=1e-4, momentum=0.9)
    
    
    scheduler_flow = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_flow, mode='min', factor=0.5, patience=3, verbose=True)
    optimizer_energy = torch.optim.SGD(model.energy_classifier.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler_energy = torch.optim.lr_scheduler.LambdaLR(
        optimizer_energy,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epoch * len(data_loader_byte_header_train_know),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / 1e-3))
        
    scaler = GradScaler()  # 使用梯度标量进行混合精度训练
    
    loss_scaler = NativeScaler()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(args.epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    center_loss = CenterLoss(num_classes=nb_classes, feat_dim=256)
    Contrast_Loss = ContrastiveLoss()
    cross_entropy_loss = nn.CrossEntropyLoss()

    T = 10   # 能量损失的温度
    for i in range(args.epoch):
        optimizer_byte_header.zero_grad()
        optimizer_byte_payload.zero_grad()
        optimizer_flow.zero_grad()
        optimizer_energy.zero_grad()
        print("特征提取模型与能量模型 第"+ str(iter_epoch) + "次 交叉迭代 第" + str(i + 1) + "轮")
        print(f"forward: Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"forward: Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        pred_all = []
        target_all = []
        total_loss = 0  # 用于累积损失
        batch_count = len(data_loader_byte_header_train_know)  # 用于记录 batch 数量
        
        iterator_know_header = iter(data_loader_byte_header_train_know)
        iterator_know_payload = iter(data_loader_byte_payload_train_know)
        iterator_unknow_header = iter(data_loader_byte_header_train_unknow)
        iterator_unknow_payload = iter(data_loader_byte_payload_train_unknow)
        data_loader_flow_train_unknown.reflesh()  # 未知类也跟着重新刷新，因为为图片数据，不能像dataloader一样

        while True:
            try:
                samples_header_train, targets_header_train = next(iterator_know_header)
                samples_payload_train, targets_payload_train = next(iterator_know_payload)
            except StopIteration:
                break
            
            try:
                samples_header_train_unknow, targets_header_train_unknow = next(iterator_unknow_header)
                samples_payload_train_unknow, targets_payload_train_unknow = next(iterator_unknow_payload)
            except StopIteration:
                iterator_unknow_header = iter(data_loader_byte_header_train_unknow)
                iterator_unknow_payload = iter(data_loader_byte_payload_train_unknow)
                samples_header_train_unknow, targets_header_train_unknow = next(iterator_unknow_header)
                samples_payload_train_unknow, targets_payload_train_unknow = next(iterator_unknow_payload)
            
            # known
            samples_header_train = samples_header_train.to(device, non_blocking=True)
            targets_header_train = targets_header_train.to(device, non_blocking=True)
            samples_payload_train = samples_payload_train.to(device, non_blocking=True)
            targets_payload_train = targets_payload_train.to(device, non_blocking=True)
            graphs_train, labels_train= data_loader_flow_train_known.next_train_batch_know(args.batch_size)
            
            # unknown
            samples_header_train_unknow = samples_header_train_unknow.to(device, non_blocking=True)
            targets_header_train_unknow = targets_header_train_unknow.to(device, non_blocking=True)
            samples_payload_train_unknow = samples_payload_train_unknow.to(device, non_blocking=True)
            targets_payload_train_unknow = targets_payload_train_unknow.to(device, non_blocking=True)
            graphs_train_unknow, labels_train_unknow= data_loader_flow_train_unknown.next_train_batch_unknow(args.batch_size)
            
            if use_gpu :
                graphs_train = graphs_train.to(torch.device(device))
                labels_train = labels_train.to(torch.device(device))
                graphs_train_unknow = graphs_train_unknow.to(torch.device(device))
                labels_train_unknow = labels_train_unknow.to(torch.device(device))
            
            if mixup_fn is not None:
                samples_header_train, targets_header_train = mixup_fn(samples_header_train, targets_header_train)
                samples_payload_train, targets_payload_train = mixup_fn(samples_payload_train, targets_payload_train)
                samples_header_train_unknow, targets_header_train_unknow = mixup_fn(samples_header_train_unknow, targets_header_train_unknow)
                samples_payload_train_unknow, targets_payload_train_unknow = mixup_fn(samples_payload_train_unknow, targets_payload_train_unknow)
                
            with torch.cuda.amp.autocast(enabled=False):
                features, outputs_in = model(samples_header_train, samples_payload_train, graphs_train, 'energy')
                features, outputs_out = model(samples_header_train_unknow, samples_payload_train_unknow, graphs_train_unknow, 'energy')
                labels_in = labels_train
                labels_out = labels_train_unknow
                labels_out_pseudo = torch.full((labels_train_unknow.shape[0], ), args.nb_classes)
            
            loss = energy.energy_ft_loss(outputs_in, outputs_out, labels_in, labels_out_pseudo)
            
            scaler.scale(loss).backward()  # 缩放损失，进行反向传播
            scaler.step(optimizer_energy)
            scheduler_energy.step()
            scaler.step(optimizer_flow)
            scheduler_flow.step(loss)
            scaler.step(optimizer_byte_payload)
            
            scaler.step(optimizer_byte_header)
            
            scaler.update()
            total_loss += loss.item()
            _, y_train_pred = torch.max(outputs_in, dim=1)
            pred_all.extend(y_train_pred.cpu().numpy().tolist())  # 转换为 NumPy 数组并追加
            target_all.extend(labels_train.cpu().numpy().tolist())  # 转换为 NumPy 数组并追加
            
        accuracy = accuracy_score(target_all, pred_all)
        precision = precision_score(target_all, pred_all, average='macro')
        recall = recall_score(target_all, pred_all, average='macro')
        f1 = f1_score(target_all, pred_all, average='macro')
        average_loss = total_loss / batch_count

        print(f"Epoch {i + 1} train Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Average Loss: {average_loss:.4f}")
        
        pred_all = []
        target_all = []
        total_loss = 0
        for data_iter_step, ((samples_header_val, targets_header_val), 
                             (samples_payload_val  , targets_payload_val)) in enumerate(
                        zip(data_loader_byte_header_val_know, 
                            data_loader_byte_payload_val_know)):
            # 获取 samples 和 targets 并移动到设备上
            samples_header_val = samples_header_val.to(device, non_blocking=True)
            targets_header_val = targets_header_val.to(device, non_blocking=True)
            
            samples_payload_val = samples_payload_val.to(device, non_blocking=True)
            targets_payload_val = targets_payload_val.to(device, non_blocking=True)
            
            graphs_val, labels_val = data_loader_flow_valid_known.next_valid_batch_know(args.batch_size)  # 数据加载器获取下一个训练批次的图和标签。
            if use_gpu :
                graphs_val = graphs_val.to(torch.device(device))
                labels_val = labels_val.to(torch.device(device))
            if mixup_fn is not None:
                samples_header_val, targets_header_val = mixup_fn(samples_header_val, targets_header_val)
                samples_payload_val, targets_payload_val = mixup_fn(samples_payload_val, targets_payload_val)
            with torch.cuda.amp.autocast(enabled=False):
                _, outputs = model(samples_header_val, samples_payload_val, graphs_val, 'energy')
            
            loss = energy.energy_loss(outputs, labels_val, T)

            total_loss += loss.item()
            _, y_val_pred = torch.max(outputs, dim=1)
            
            pred_all.extend(y_val_pred.cpu().numpy().tolist())  # 转换为 NumPy 数组并追加
            target_all.extend(labels_val.cpu().numpy().tolist())  # 转换为 NumPy 数组并追加
            
        accuracy = accuracy_score(target_all, pred_all)
        precision = precision_score(target_all, pred_all, average='macro')
        recall = recall_score(target_all, pred_all, average='macro')
        f1 = f1_score(target_all, pred_all, average='macro')
        average_loss = total_loss / batch_count
        # 打印评估指标
        print()
        print(f"Epoch {i + 1} valid Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Average Loss: {average_loss:.4f}")
        print("-" * 100)

# ODC_iteration
def train_iteration_ODC_ft(model, nb_classes, use_gpu, device, dataset_name,
        data_loader_byte_header_train_unknow,
        data_loader_byte_payload_train_unknow,
        data_loader_flow_train_unknown, is_init_odc, iter_epoch, ratio, new_classes):
    for param in model.byte_header_extractor.parameters():
        param.requires_grad = True
    for param in model.byte_payload_extractor.parameters():
        param.requires_grad = True
    for param in model.flow_extractor.parameters():
        param.requires_grad = True
    for param in model.energy_classifier.parameters():
        param.requires_grad = False
    for param in model.odc_classifier.parameters():
        param.requires_grad = True
    graphs_train_unknown = []
    labels_train_unknown = []
    data_loader_flow_train_unknown.reflesh()  # 未知类也跟着重新刷新，从头开始，防止出现从某一位置开始遍历
    for i in data_loader_flow_train_unknown.unknow_train_index:
        graphs_train_unknown.append(data_loader_flow_train_unknown.graphs[i])
        labels_train_unknown.append(data_loader_flow_train_unknown.labelId[i])
    feature_flow_train_unknown = dgl.batch(graphs_train_unknown)
    labels_flow_train_unknown = torch.tensor(labels_train_unknown)
    save_dataloader_to_h5(data_loader_byte_header_train_unknow, 'dataset/dataloader/' + dataset_name + '_data_loader_byte_header_train_unknow.h5')
    save_dataloader_to_h5(data_loader_byte_payload_train_unknow, 'dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_train_unknow.h5')
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_train_unknow.h5', 'r') as h5f:
        feature_byte_header_train_unknow = torch.tensor(h5f['features'])
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_train_unknow.h5', 'r') as h5f:
        feature_byte_payload_train_unknow = torch.tensor(h5f['features'])
    features_all, outputs = model(feature_byte_header_train_unknow.to(device), feature_byte_payload_train_unknow.to(device), feature_flow_train_unknown.to(device), 'energy')
    features_all = features_all.detach()
    
    args = get_args_parser_byte()
    args = args.parse_args()
    args.nb_classes = int(nb_classes * ratio)
    args.epoch = 10
    seed = args.seed + misc.get_rank()  # 生成随机种子，使结果具有可复现性
    torch.manual_seed(seed)   # 设置PyTorch的随机种子
    np.random.seed(seed)      # 设置NumPy的随机种子
    cudnn.benchmark = True    # 使CuDNN在不同输入大小时找到最佳内核，提高计算效率
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
        
    model_without_ddp_header = model.byte_header_extractor
    model_without_ddp_payload = model.byte_payload_extractor
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    
    param_groups_header = lrd.param_groups_lrd(model_without_ddp_header, args.weight_decay,
        no_weight_decay_list=model_without_ddp_header.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    
    param_groups_payload = lrd.param_groups_lrd(model_without_ddp_payload, args.weight_decay,
        no_weight_decay_list=model_without_ddp_payload.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    
    optimizer_byte_header = torch.optim.SGD(param_groups_header, lr=1e-4, momentum=0.9)
    optimizer_byte_payload = torch.optim.SGD(param_groups_payload, lr=1e-4, momentum=0.9)
    optimizer_flow = torch.optim.SGD(model.flow_extractor.parameters(), lr=1e-4, momentum=0.9)
    scheduler_flow = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_flow, mode='min', factor=0.5, patience=3, verbose=True)
    optimizer_odc_classifier = torch.optim.SGD(model.odc_classifier.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler_odc_classifier = torch.optim.lr_scheduler.LambdaLR(
        optimizer_odc_classifier,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epoch * len(data_loader_byte_header_train_unknow),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / 1e-3))
    scaler = GradScaler()  # 使用梯度标量进行混合精度训练
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # model.odc_classifier.initialize_memory_bank(features_all, num_clusters=(int(nb_classes * (1 - ratio)) - new_classes)) # 未知类别数目
    # if is_init_odc == False:
    model.odc_classifier.initialize_memory_bank(features_all, num_clusters= nb_classes - int(nb_classes * ratio)) # 未知类别数目
    for i in range(args.epoch):
        optimizer_byte_header.zero_grad()
        optimizer_byte_payload.zero_grad()
        optimizer_flow.zero_grad()
        optimizer_odc_classifier.zero_grad()
        print("第"+ str(iter_epoch) + "次 交叉迭代 第" + str(i + 1) + "轮")
        print(f"forward: Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"forward: Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_change_ratio = 0.0
        total_samples = 0
        for data_iter_step, ((samples_header_train, targets_header_train), 
                    (samples_payload_train  , targets_payload_train)) in enumerate(
            zip(data_loader_byte_header_train_unknow, 
                data_loader_byte_payload_train_unknow)):
            samples_header_train = samples_header_train.to(device, non_blocking=True)
            targets_header_train = targets_header_train.to(device, non_blocking=True)
            samples_payload_train = samples_payload_train.to(device, non_blocking=True)
            targets_payload_train = targets_payload_train.to(device, non_blocking=True)
            
            graphs_train, labels_train = data_loader_flow_train_unknown.next_train_batch_unknow(args.batch_size)  # 数据加载器获取下一个训练批次的图和标签。
            if use_gpu :
                graphs_train = graphs_train.to(torch.device(device))
                labels_train = labels_train.to(torch.device(device))
            if mixup_fn is not None:
                samples_header_train, targets_header_train = mixup_fn(samples_header_train, targets_header_train)
                samples_payload_train, targets_payload_train = mixup_fn(samples_payload_train, targets_payload_train)
            with torch.cuda.amp.autocast(enabled=False):
                features, cls_score = model(samples_header_train, samples_payload_train, graphs_train, 'odc')

            idx = torch.arange(data_iter_step * args.batch_size, data_iter_step * args.batch_size + len(features))

            pseudo_labels = model.odc_classifier.memory_bank['labels'][idx].clone().detach().to(dtype=torch.long).to(cls_score[0].device)
            losses = model.odc_classifier.loss(features, model.odc_classifier.memory_bank['centroids'], cls_score, pseudo_labels) 
            change_ratio = model.odc_classifier.update_memory_bank_samples(idx, features.detach())
            losses['change_ratio'] = change_ratio
            loss = model.odc_classifier.parse_losses(losses)
            loss.backward()
            optimizer_odc_classifier.step()
            scheduler_odc_classifier.step()
            optimizer_flow.step()
            scheduler_flow.step(loss)
            optimizer_byte_payload.step()
            optimizer_byte_header.step()
            optimizer_byte_header.zero_grad()
            optimizer_byte_payload.zero_grad()
            optimizer_flow.zero_grad()
            optimizer_odc_classifier.zero_grad()
            
            epoch_loss += losses['loss'].detach().item()   # 累加每个batch的损失
            epoch_accuracy += losses['acc'] * features.size(0)  # 累加每个batch的准确率
            epoch_change_ratio += losses['change_ratio'] * features.size(0)
            total_samples += features.size(0)  # 累加总样本数
            model.odc_classifier.update_centroids()
            model.odc_classifier.deal_with_small_clusters()
    
        avg_loss = torch.tensor(epoch_loss, dtype = torch.float32, requires_grad=True)
        (avg_loss / len(data_loader_byte_header_train_unknow)).backward()
        
        optimizer_odc_classifier.step()
        scheduler_odc_classifier.step()
        optimizer_flow.step()
        scheduler_flow.step(loss)
        optimizer_byte_payload.step()
        optimizer_byte_header.step()

        new_labels = model.odc_classifier.memory_bank['labels']
        if new_labels.is_cuda:
            new_labels = new_labels.cpu()
        histogram = np.bincount(new_labels, minlength=nb_classes - int(nb_classes * ratio))
        non_empty_clusters = np.count_nonzero(histogram)

        current_lr = scheduler_odc_classifier.get_last_lr()[0]
        avg_accuracy = epoch_accuracy / total_samples
        avg_change_ratio = epoch_change_ratio / total_samples
        print(f"Epoch [{i+1}/{args.epoch}]: lr: {current_lr}, loss: {(avg_loss / len(data_loader_byte_header_train_unknow))}, change_ratio: {avg_change_ratio}, acc:{avg_accuracy}")
        print("-" * 100)

# train
def train(model, nb_classes, use_gpu, device, dataset_name, 
        data_loader_byte_header_train_know, data_loader_byte_header_val_know,
        data_loader_byte_payload_train_know, data_loader_byte_payload_val_know,
        data_loader_flow_train_known, data_loader_flow_valid_known,
        data_loader_byte_header_train_unknow,
        data_loader_byte_payload_train_unknow,
        data_loader_flow_train_unknown,
        ratio, new_classes):
    torch.manual_seed(256)
    np.random.seed(256)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cudnn.benchmark = True  # 使CuDNN在不同输入大小时找到最佳内核，提高计算效率
    epoch = 10
    model.to(device)
    model.train(True)
    model.odc_classifier.init_weights()
    is_init_odc = False
    # klnd
    MAVs = [None] * int(nb_classes * ratio)
    thresholds = [None] * int(nb_classes * ratio)

    for i in range(epoch):
        if i == 0:
            if not os.path.exists('dataset/saved_models/' + dataset_name + 
                '_Pre_FEC_OSL_model_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.pth'):
                if os.path.exists('dataset/saved_models/' + dataset_name + 
                    '_byte_header_extractor_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.pth'):
                    model.byte_header_extractor.load_state_dict(torch.load('dataset/saved_models/' + dataset_name + 
                        '_byte_header_extractor_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.pth'))
                else:
                    train_header_model(model, int(nb_classes * ratio), device,
                        data_loader_byte_header_train_know, data_loader_byte_header_val_know,
                        dataset_name, nb_classes)
                
                if os.path.exists('dataset/saved_models/' + dataset_name + 
                    '_byte_payload_extractor_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.pth'):
                    model.byte_payload_extractor.load_state_dict(torch.load('dataset/saved_models/' + dataset_name + 
                        '_byte_payload_extractor_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.pth'))
                else:
                    train_payload_model(model, int(nb_classes * ratio), device,
                        data_loader_byte_payload_train_know, data_loader_byte_payload_val_know,
                        dataset_name, nb_classes)
                    
                if os.path.exists('dataset/saved_models/' + dataset_name + 
                    '_flow_extractor_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.pth'):
                    model.flow_extractor.load_state_dict(torch.load('dataset/saved_models/' + dataset_name + 
                        '_flow_extractor_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.pth'))
                else:
                    # data_loader_flow_train_known, data_loader_flow_valid_known,
                    train_flow_model(model, use_gpu, device, data_loader_flow_train_known, data_loader_flow_valid_known, 
                        dataset_name, nb_classes, int(nb_classes * ratio))
                    
                Pre_train_ft(model, nb_classes, use_gpu, device, dataset_name,
                    data_loader_byte_header_train_know, data_loader_byte_header_val_know,
                    data_loader_byte_payload_train_know, data_loader_byte_payload_val_know, 
                    data_loader_flow_train_known, data_loader_flow_valid_known,
                    data_loader_byte_header_train_unknow,
                    data_loader_byte_payload_train_unknow,
                    data_loader_flow_train_unknown, ratio)
                
            else:
                # print(dataset_name + '_Pre_FEC_OSL_model 已存在   正在加载...')
                model = torch.load('dataset/saved_models/' + dataset_name + '_Pre_FEC_OSL_model_' + \
                    str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.pth')
                model.to(device)
                model.byte_header_extractor.to(device)
                model.byte_payload_extractor.to(device)
                model.flow_extractor.to(device)
                model.energy_classifier.to(device)
                model.odc_classifier.to(device)
                model.train(True)
                model.byte_header_extractor.train(True)
                model.byte_payload_extractor.train(True)
                model.flow_extractor.train(True)
                model.energy_classifier.train(True)
                model.odc_classifier.train(True)
                
            is_init_odc = True
            
        print("Start Cross-iteration")
        print("-" * 100)
        print("训练第" + str(i + 1) + "轮")
        print("-" * 100)
        print()
        
        train_iteration_ODC_ft(model, nb_classes, use_gpu, device, dataset_name,
            data_loader_byte_header_train_unknow,
            data_loader_byte_payload_train_unknow,
            data_loader_flow_train_unknown, is_init_odc, i + 1, ratio, new_classes)
        train_iteration_energy_ft(model, nb_classes, use_gpu, device,
            data_loader_byte_header_train_know, data_loader_byte_header_val_know,
            data_loader_byte_payload_train_know, data_loader_byte_payload_val_know, 
            data_loader_flow_train_known, data_loader_flow_valid_known,
            data_loader_byte_header_train_unknow,
            data_loader_byte_payload_train_unknow,
            data_loader_flow_train_unknown, i + 1, ratio, MAVs, thresholds)

    save_path = 'dataset/saved_models/' + dataset_name + '_FEC_OSL_model_16_4_cross.pth'
    torch.save(model, save_path)
    print("成功保存" + "model" + "模型到：" + 'dataset/saved_models/' + dataset_name + '_FEC_OSL_model_16_4_cross.pth')
    return model

# merge and shuffle functions
def merge_and_shuffle(features_list, labels_list):
    torch.manual_seed(256)
    np.random.seed(256)
    merged_features = torch.cat(features_list, dim=0)
    merged_labels = torch.cat(labels_list, dim=0)
    
    num_samples = merged_features.shape[0]
    indices = torch.randperm(num_samples)
    
    shuffled_features = merged_features[indices]
    shuffled_labels = merged_labels[indices]
    
    return shuffled_features, shuffled_labels

def merge_and_shuffle_graphs(graphs_list, labels_list):
    torch.manual_seed(256)
    np.random.seed(256)
    merged_graphs = []
    merged_labels = []
    
    for graph_list in graphs_list:
        merged_graphs.append(graph_list)
    for label_list in labels_list:
        merged_labels.append(label_list)
    
    num_samples = len(merged_labels)
    indices = torch.randperm(num_samples)
    
    shuffled_graphs = [merged_graphs[idx] for idx in indices]
    shuffled_labels = torch.tensor([merged_labels[idx] for idx in indices])
    
    return shuffled_graphs, shuffled_labels

# test
def test(model, nb_classes, use_gpu, device, dataset_name, 
        data_loader_byte_header_train_know, data_loader_byte_header_valid_know,
        data_loader_byte_payload_train_know, data_loader_byte_payload_valid_know, 
        data_loader_flow_train_known, data_loader_flow_valid_known,
        data_loader_byte_header_valid_unknow,
        data_loader_byte_payload_valid_unknow,
        data_loader_flow_valid_unknown,
        ratio):
    for param in model.byte_header_extractor.parameters():
        param.requires_grad = False
    for param in model.byte_payload_extractor.parameters():
        param.requires_grad = False
    for param in model.flow_extractor.parameters():
        param.requires_grad = False
    for param in model.energy_classifier.parameters():
        param.requires_grad = False
    for param in model.odc_classifier.parameters():
        param.requires_grad = False
    model.to(device)
    model.byte_header_extractor.to(device)
    model.byte_payload_extractor.to(device)
    model.flow_extractor.to(device)
    model.energy_classifier.to(device)
    model.odc_classifier.to(device)
    
    model.eval()  # 设置模型为评估模式
    model.byte_header_extractor.eval()
    model.byte_payload_extractor.eval()
    model.flow_extractor.eval()
    model.energy_classifier.eval()
    model.odc_classifier.eval()
    
    save_dataloader_to_h5(data_loader_byte_header_train_know, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_header_train_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5')
    save_dataloader_to_h5(data_loader_byte_header_valid_know, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_header_valid_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5')
    save_dataloader_to_h5(data_loader_byte_header_valid_unknow, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_header_valid_unknow_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5')
    
    save_dataloader_to_h5(data_loader_byte_payload_train_know, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_payload_train_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5')
    save_dataloader_to_h5(data_loader_byte_payload_valid_know, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_payload_valid_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5')
    save_dataloader_to_h5(data_loader_byte_payload_valid_unknow, 'dataset/dataloader/' + dataset_name + 
        '_data_loader_byte_payload_valid_unknow_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5')
    graphs_train_known = []
    labels_train_known = []
    graphs_valid_known = []
    labels_valid_known = []
    graphs_valid_unknown = []
    labels_valid_unknown = []

    for i in data_loader_flow_train_known.know_train_index:
        graphs_train_known.append(data_loader_flow_train_known.graphs[i])
        labels_train_known.append(data_loader_flow_train_known.labelId[i])
    for i in data_loader_flow_valid_known.know_valid_index:
        graphs_valid_known.append(data_loader_flow_valid_known.graphs[i])
        labels_valid_known.append(data_loader_flow_valid_known.labelId[i])
    for i in data_loader_flow_valid_unknown.unknow_valid_index:
        graphs_valid_unknown.append(data_loader_flow_valid_unknown.graphs[i])
        labels_valid_unknown.append(data_loader_flow_valid_unknown.labelId[i])
    graphs_weibull_known = dgl.batch(graphs_train_known)
    labels_weibull_known = torch.tensor(labels_train_known)
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_train_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5', 'r') as h5f:
        feature_byte_header_weibull_know = torch.tensor(h5f['features'])
        label_byte_header_weibull_know = torch.tensor(h5f['labels'])
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_train_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5', 'r') as h5f:
        feature_byte_payload_weibull_know = torch.tensor(h5f['features'])
        label_byte_payload_weibull_know = torch.tensor(h5f['labels'])
    
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_valid_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5', 'r') as h5f:
        feature_byte_header_valid_know = torch.tensor(h5f['features'])
        label_byte_header_valid_know = torch.tensor(h5f['labels'])
        
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_valid_know_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5', 'r') as h5f:
        feature_byte_payload_valid_know = torch.tensor(h5f['features'])
        label_byte_payload_valid_know = torch.tensor(h5f['labels'])
    
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_header_valid_unknow_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5', 'r') as h5f:
        feature_byte_header_valid_unknow = torch.tensor(h5f['features'])
        label_byte_header_valid_unknow = torch.tensor(h5f['labels'])
        # print(feature_byte_header_valid_unknow.shape)
    with h5py.File('dataset/dataloader/' + dataset_name + '_data_loader_byte_payload_valid_unknow_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) + '.h5', 'r') as h5f:
        feature_byte_payload_valid_unknow = torch.tensor(h5f['features'])
        label_byte_payload_valid_unknow = torch.tensor(h5f['labels'])
        
    feature_byte_header_valid, label_byte_header_valid = \
        merge_and_shuffle([feature_byte_header_valid_know, feature_byte_header_valid_unknow],
                        [label_byte_header_valid_know, label_byte_header_valid_unknow])
    feature_byte_payload_valid, label_byte_payload_valid = \
        merge_and_shuffle([feature_byte_payload_valid_know, feature_byte_payload_valid_unknow],
                        [label_byte_payload_valid_know, label_byte_payload_valid_unknow])
    
    flow_graphs_valid, flow_labels_valid = merge_and_shuffle_graphs(
        graphs_valid_known + graphs_valid_unknown,
        labels_valid_known +  labels_valid_unknown
    )
    
    T = 10
    threshold = 0.05
    with torch.no_grad():
        features_known, weibull_logits = model(feature_byte_header_weibull_know.to(device),
            feature_byte_payload_weibull_know.to(device),
            graphs_weibull_known.to(device), 'energy')
        know_energy = energy.calculate_energy(weibull_logits, T).cpu().numpy()
        
    shape, loc, scale = energy.weibull_min.fit(-know_energy)  # Weibull 分布的形状（shape）、位置（loc）和尺度（scale）参数用于描述数据的分布特性。
    
    labels_valid = flow_labels_valid
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            feature_matrix, logits = model(feature_byte_header_valid.to(device), feature_byte_payload_valid.to(device), dgl.batch(flow_graphs_valid).to(device), 'energy')

            feature_matrix_known, logits_known = model(feature_byte_header_weibull_know.to(device), feature_byte_payload_weibull_know.to(device), dgl.batch(graphs_train_known).to(device), 'energy')
            test_energy = energy.calculate_energy(logits, T).detach().cpu().numpy()
            cdf_values = weibull_min.cdf(-test_energy, shape, loc, scale)
            tau = -weibull_min.ppf(0.05, shape, loc, scale)
            print("energy_τ = ", tau)
            test_mask = cdf_values < threshold
            test_tensor = torch.tensor(test_mask, dtype=torch.bool)
            
            # 区分已知/未知
            test_known_data = feature_matrix[~test_tensor].to(device)
            test_unknown_data = feature_matrix[test_tensor].to(device)
            test_known_label = labels_valid[~test_tensor].to(device)
            test_unknown_label = labels_valid[test_tensor].to(device)
            
                        
            # 拟合已知类和未知类的 Weibull 分布
            with torch.no_grad():
                known_inputs = torch.tensor(test_known_data, dtype=torch.float32).to(device)
                known_logits = model.energy_classifier(known_inputs)
                known_energy = energy.calculate_energy(known_logits, T).cpu().numpy()

                unknown_inputs = torch.tensor(test_unknown_data, dtype=torch.float32).to(device)
                unknown_logits = model.energy_classifier(unknown_inputs)
                unknown_energy = energy.calculate_energy(unknown_logits, T).cpu().numpy()
            known_params = weibull_min.fit(-known_energy)
            unknown_params = weibull_min.fit(-unknown_energy)

            # 绘制已知类和未知类的 Weibull 分布在同一张图上
            energy.plot_two_weibull(
                known_energy=known_energy, 
                known_params=known_params,
                unknown_energy=unknown_energy, 
                unknown_params=unknown_params,
                title='Weibull',
                save_path='graph/weibull_comparison.png'
            )
            
            # 已知类测试
            byte_header_valid_know = feature_byte_header_valid[~test_tensor]
            byte_payload_valid_know = feature_byte_payload_valid[~test_tensor]
            
            graphs_valid_know = [value for value, condition in zip(flow_graphs_valid, test_tensor) if not condition]
            
            _, outputs = model(feature_byte_header_valid_know.to(device), feature_byte_payload_valid_know.to(device), dgl.batch(graphs_valid_known).to(device), 'energy')
            _, y_test_pred = torch.max(outputs, dim=1)
            test_known_label_list = label_byte_header_valid_know.cpu().numpy()
            y_test_pred_list = y_test_pred.cpu().numpy()
            
            accuracy = accuracy_score(test_known_label_list, y_test_pred_list)
            precision = precision_score(test_known_label_list, y_test_pred_list, average='macro')
            recall = recall_score(test_known_label_list, y_test_pred_list, average='macro')
            f1 = f1_score(test_known_label_list, y_test_pred_list, average='macro')

            print(f"Epoch test Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("-" * 100)
            

            y_test = labels_valid.tolist()
            y_unknown_detected = test_unknown_label.tolist()
            count_dict_test = {}
            for num in y_test:
                if num in count_dict_test:
                    count_dict_test[num] += 1
                else:
                    count_dict_test[num] = 1

            count_dict_unknown = {}
            for num in y_unknown_detected:
                if num in count_dict_unknown:
                    count_dict_unknown[num] += 1
                else:
                    count_dict_unknown[num] = 1
            
            know_list = [i for i in range(int(nb_classes * ratio))]  # 已知类别标签数组
            unknow_list = [i for i in range(int(nb_classes * ratio), nb_classes)]  # 未知类别标签数组
            TP = sum(count_dict_unknown.get(cls, 0) for cls in unknow_list)  # 真正例：正确检测的未知类
            FP = sum(count_dict_unknown.get(cls, 0) for cls in know_list)  # 假正例：已知类被误判为未知类

            FN = sum(count_dict_test.get(cls, 0) for cls in unknow_list) - TP  # 假负例：未知类未被检测
            TN = sum(count_dict_test.get(cls, 0) for cls in know_list) - FP  # 真负例：已知类正确检测
            
            accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')
            y_true = [1] * TP + [0] * TN + [1] * FN + [0] * FP  # TP + TN + FN + FP
            y_scores = [1] * TP + [0] * TN + [0] * FN + [1] * FP  # TP的预测概率为1，TN为0，FN预测为0，FP为1

            auc_score = energy.roc_auc_score(y_true, y_scores)
            print(f'AUC: {auc_score:.4f}')
            
            
            # cluster
            inputs = test_unknown_data.to(device)
            output = model.odc_classifier(inputs)
            _, predicted = torch.max(output[0], 1)
            true_labels = test_unknown_label
            cluster_labels_numpy = torch.tensor(predicted).cpu().numpy()
            true_labels_numpy = true_labels.clone().detach().cpu().numpy()
            ami_score = adjusted_mutual_info_score(true_labels_numpy, cluster_labels_numpy)
            print(f'Test ami_score: {ami_score:.4f}')  # 输出测试结果
                            
def main():
    nb_classes = get_classes("dataset/pcap/USTC-TFC2016_change_PCAP")
    ratio = 0.8
    new_classes = math.ceil(nb_classes * ratio / 3)  # 新类个数
    
    # USTC-TFC2016
    flows_pcap_path = "dataset/pcap/USTC-TFC2016_PCAP"
    if not os.path.exists("dataset/byte_data/USTC-TFC2016_" + 
        str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio))):
        os.makedirs("dataset/byte_data/USTC-TFC2016_" + 
            str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)))
    output_header_path = "dataset/byte_data/USTC-TFC2016_" + \
        str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) +  "/header"
    output_payaload_path = "dataset/byte_data/USTC-TFC2016_" + \
        str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) +  "/payload"
    
    data_process_header.Grayscale_Image_generator(flows_pcap_path, output_header_path)
    data_process_payload.Grayscale_Image_generator(flows_pcap_path, output_payaload_path)
    
    output_flow_path = "dataset/flow_data/USTC-TFC2016_JSON_" + \
        str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio))
    pacp_to_json.to_json(flows_pcap_path, output_flow_path, nb_classes, int(nb_classes * ratio))
    

    dataset_name = "USTC-TFC2016"
    dataset_header_path = "dataset/byte_data/USTC-TFC2016_" + \
        str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) +  "/header"  # USTC-TFC2016
    dataset_payload_path = "dataset/byte_data/USTC-TFC2016_" + \
        str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio)) +  "/payload"  # USTC-TFC2016
    dataset_flow_path = 'USTC-TFC2016_JSON_' + str(int(nb_classes * ratio)) + "_" + str(nb_classes - int(nb_classes * ratio))
    
    use_gpu = torch.cuda.is_available()
    device_id = select_gpu.get_free_gpu_id()
    if use_gpu :
        device= "cuda:{0}".format(device_id)
    else:
        device= "cpu"
    device = "cuda:3"
    model = FECOSLModel(nb_classes, use_gpu, device, ratio, new_classes)

    data_loader_byte_header_train_know,\
    data_loader_byte_header_valid_know,\
    data_loader_byte_header_train_unknow,\
    data_loader_byte_header_valid_unknow = get_byte_header_dataset(dataset_header_path, ratio, new_classes)
    
    data_loader_byte_payload_train_know,\
    data_loader_byte_payload_valid_know,\
    data_loader_byte_payload_train_unknow,\
    data_loader_byte_payload_valid_unknow = get_byte_payload_dataset(dataset_payload_path, ratio, new_classes)
    
    data_loader_flow_train_known,\
    data_loader_flow_valid_known,\
    data_loader_flow_train_unknown,\
    data_loader_flow_valid_unknown = get_flow_dataset(dataset_flow_path, nb_classes, int(nb_classes * ratio))
    
    # train
    model = train(model, nb_classes, use_gpu, device, dataset_name,
        data_loader_byte_header_train_know, data_loader_byte_header_valid_know,
        data_loader_byte_payload_train_know, data_loader_byte_payload_valid_know, 
        data_loader_flow_train_known, data_loader_flow_valid_known,
        data_loader_byte_header_train_unknow,
        data_loader_byte_payload_train_unknow,
        data_loader_flow_train_unknown,
        ratio, new_classes
    )
    # model = torch.load('dataset/saved_models/USTC-TFC2016_Pre_FEC_OSL_model_16_4_cross.pth')
    # model = torch.load('dataset/saved_models/USTC-TFC2016_Pre_FEC_OSL_model_16_4_T_test.pth')
    
    # test
    test(model, nb_classes, use_gpu, device, dataset_name,
        data_loader_byte_header_train_know, data_loader_byte_header_valid_know,
        data_loader_byte_payload_train_know, data_loader_byte_payload_valid_know, 
        data_loader_flow_train_known, data_loader_flow_valid_known,
        data_loader_byte_header_valid_unknow,
        data_loader_byte_payload_valid_unknow,
        data_loader_flow_valid_unknown,
        ratio
    )
    
if __name__ == '__main__':
    main()