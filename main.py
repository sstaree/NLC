# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import random
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
from model import DICNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import scipy.io
from sklearn.preprocessing import OneHotEncoder
import h5py
import math
from loss import Loss
import pickle
import matplotlib.pyplot as plt
# import os
import pandas as pd
import csv
from measure import *
parser = argparse.ArgumentParser()

# parser.add_argument('--lrkl', type=float, default=0.05)
# parser.add_argument('--lrkl1', type=float, default=0.001)
parser.add_argument('--lrkl', type=float, default=0.001)       #0.001 for UCI,Caltech101  0.05 for Leaves Scene15 LandUse21 ALOI
parser.add_argument('--lrkl1', type=float, default=0.001)
parser.add_argument('--Nlabel', default=7, type=int)
parser.add_argument('--Nz', default=7, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--warm_epoch', default=20,type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--MaskRatios', type=float, default=0.5)
parser.add_argument('--LabelMaskRatio', type=float, default=0.5)
parser.add_argument('--TraindataRatio', type=float, default=0.8)
parser.add_argument('--AE_shuffle', type=bool, default=True)
parser.add_argument('--min_AP', default=0., type=float)
parser.add_argument('--tol', default=1e-7, type=float)
parser.add_argument('--noiseRatio', default=0, type=float)
parser.add_argument('--lambda_epochs', type=int, help='gradually increase the value of lambda from 0 to 1', default=100)

parser.add_argument('--dataset', type=str, help='PIE, Caltech101, UCI, BBC, Leaves Scene LandUse_21',
                    default="PIE")  # Due to file size limitations, we have only uploaded the PIE, Leaves, and UCI datasets here
parser.add_argument('--noise_type', type=str, help="sym, flip, IDN", default="IDN")
parser.add_argument('--beta', type=float, default=1e-3)    #ranking loss
parser.add_argument('--alpha', type=float, default=0.7)
parser.add_argument('--momentumkl', type=float, default=0.9)
parser.add_argument('--num_neighbor', type=int, default=20)
parser.add_argument('--threshold_sver', type=float, default=0.45)
parser.add_argument('--threshold_scor', type=float, default=0.6)
parser.add_argument('--knn_threshold', type=float, default=0)
parser.add_argument('--high_scor', default=0.95, type=float)
parser.add_argument('--num_class', default=1.0, type=float)
parser.add_argument('--correct', default=True, type=bool)
parser.add_argument('--mixup', default=True, type=bool)

args = parser.parse_args()

args.data_path = './Datasets/' + args.dataset


class My_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, zp_pro_0, p0):
        r = 0.5
        sf = nn.Softmax(dim=1)
        zp_pro_0 = zp_pro_0.float()
        p0 = p0.float()
        zp_pro_0 = sf(zp_pro_0)  # 模型预测的zp_pro_0，转化为概率
        res = torch.mm(zp_pro_0, p0.t())  #
        res = res.diag()
        res = ((1 - res ** r) / r).mean()
        return res


class My_BCE_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, zp_pro_0, p0):
        r = 0.5
        sf = nn.Softmax(dim=1)
        zp_pro_0 = zp_pro_0.float()
        p0 = p0.float()
        zp_pro_0 = sf(zp_pro_0)  # 模型预测的zp_pro_0，转化为概率
        res = torch.mm(zp_pro_0, p0.t())  #
        res = res.diag()
        res = res.mean()
        return res


def set_seed(seed=2000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mseLoss(pred, target, model):
    loss = nn.CrossEntropyLoss(reduction='mean')
    regularization_loss = 0  # 正则项
    for param in model.parameters():
        regularization_loss += torch.sum(param ** 2)  # 计算所有参数平方
    return loss(pred, target) + 0.01 * regularization_loss  # 返回损失。


def newLoss(f, s):
    r = 0.5
    return (1. - (s * f).sum(1) ** r) / r


def import_and_load_data(mu):
    # load data
    with open(args.data_path + '/' + args.noise_type + "-{:.2f}".format(mu) + "/X.pkl", 'rb') as file:
        X = pickle.load(file)
    with open(args.data_path + '/' + args.noise_type + "-{:.2f}".format(mu) + "/Y_true.pkl", 'rb') as file:
        Y_true = pickle.load(file)
    with open(args.data_path + '/' + args.noise_type + "-{:.2f}".format(mu) + "/Y_noisy.pkl", 'rb') as file:
        Y_noisy = pickle.load(file)
    return X, Y_true, Y_noisy


# loss = nn.CrossEntropyLoss()
loss = mseLoss
loss1 = My_loss()
loss2 = My_BCE_loss()


def train_DIC(mul_X, mul_X_val, WE, WE_val, yt_label_noisy, yv_label, yt_label_clean, yv_label_clean, device, args):
    # return None, torch.randn(9, 1)
    model = DICNet(
        n_stacks=4,
        n_input=args.n_input,
        n_z=args.N_z,
        Nlabel=args.Nlabel).to(device)
    loss_model = Loss(args.alpha, device)
    num_label = yv_label.shape[1]
    args.num_class = num_label
    yt_label_correct = yt_label_noisy.copy()
    mul_X_1 = mul_X
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Module):
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    nn.init.constant_(mm.bias, 0.0)

    num_X = mul_X[0].shape[0]  # 样本数量
    optimizer = SGD(model.parameters(), lr=args.lrkl, momentum=args.momentumkl)
    optimizer1 = SGD(model.parameters(), lr=args.lrkl1, momentum=args.momentumkl)
    plt_val_acc = []
    plt_train_acc = []
    plt_val_loss = []
    plt_train_loss = []
    plt_noise_num = []
    plt_clean_num = []
    plt_axix = []
    for epoch in range(int(args.epochs)):
        model.train()
        index_array = np.arange(num_X)
        if args.AE_shuffle == True:
            np.random.shuffle(index_array)
        noisy_label = np.argmax(yt_label_correct, axis=1)
        true_label = np.argmax(yt_label_clean, axis=1)
        num_e = np.sum(np.equal(noisy_label, true_label))
        num_ne = np.sum(np.not_equal(noisy_label, true_label))
        if epoch<args.warm_epoch:
            train_loss,train_acc = train(epoch,mul_X,yt_label_clean,yt_label_correct,args,model,device,optimizer,loss_model,loss,WE,index_array)
        else:
            prob, calibrated_labels,pseudo_noise_ratio = eval_train_nce(model, device,mul_X, WE,np.argmax(yt_label_correct,axis=1),args.batch_size,args.num_class,args,args.num_neighbor)

            if args.correct:
                calibrated_labels = torch.tensor(calibrated_labels)
                given_labels = torch.full(size=(num_X, num_label), fill_value=0, dtype=float)
                given_labels.scatter_(dim=1, index=torch.unsqueeze(calibrated_labels, dim=1), value=1.0)
                yt_label_correct = given_labels.numpy()
            train_loss,train_acc = train(epoch,mul_X_1,yt_label_clean,yt_label_correct,args,model,device,optimizer1,loss_model,loss,WE,index_array)


        plt_train_loss.append(train_loss)
        plt_train_acc.append(train_acc)
        val_loss,val_acc = test(epoch,mul_X_val,yv_label_clean,yv_label,args,model,device,optimizer,loss_model,loss,WE,index_array)
        plt_val_loss.append(val_loss)
        plt_val_acc.append(val_acc)
        if (epoch+1)%10==0:
            plt_noise_num.append(num_ne)
            plt_clean_num.append(num_e)
            plt_axix.append(str(epoch+1))
    return plt_train_acc[-1] * 100, plt_val_acc[-1] * 100





def main():
    global args
    args.cuda = torch.cuda.is_available()
    # print("use cuda: {}".format(args.cuda))

    device = torch.device("cuda" if args.cuda else "cpu")

    X, Y_clean, Y_noisy = import_and_load_data(args.noiseRatio)

    view_num = len(X)
    label = Y_noisy
    label_clean = Y_clean
    # print(label.shape)
    mul_X = [None] * view_num

    folds_data = np.ones((X[0].shape[0], len(X)))
    folds_label = np.ones((Y_noisy.shape[0], Y_noisy.shape[1]))

    indexstart = [k for k in range(label.shape[0])]
    random.shuffle(indexstart)

    folds_sample_index = np.array([indexstart])
    random.shuffle(folds_sample_index)

    Ndata, args.Nlabel = label.shape
    args.N_z = args.Nlabel

    indexperm = folds_sample_index

    # indexperm = folds_sample_index
    train_num = math.ceil(Ndata * args.TraindataRatio)

    train_index = indexperm[0,
                  0:train_num] - 1  # matlab generates the index from '1' to 'Nsample', but python needs from '0' to 'Nsample-1'
    remain_num = Ndata - train_num
    # val_num = math.ceil(remain_num*0.5)
    val_num = remain_num
    # print("val_num:", val_num)

    val_index = indexperm[0, train_num:train_num + val_num] - 1

    WE = folds_data  # 不完整视图指示矩阵

    # exit()

    # incomplete label construction
    obrT = folds_label  # incomplete label index

    # exit()
    if label.min() == -1:
        label = (label + 1) * 0.5
    Inc_label = label * obrT  # incomplete label matrix
    fan_Inc_label = 1 - Inc_label
    # incomplete data construction
    for iv in range(view_num):
        mul_X[iv] = np.copy(X[iv])
        # mul_X[iv] = torch.tensor(mul_X[iv])
        # mul_X[iv] = F.normalize(mul_X[iv])
        # mul_X[iv] = mul_X[iv].numpy()
        mul_X[iv] = mul_X[iv].astype(np.float32)
        WEiv = WE[:, iv]
        ind_1 = np.where(WEiv == 1)
        ind_1 = (np.array(ind_1)).reshape(-1)
        ind_0 = np.where(WEiv == 0)
        ind_0 = (np.array(ind_0)).reshape(-1)
        mul_X[iv][ind_1, :] = StandardScaler().fit_transform(mul_X[iv][ind_1, :])
        mul_X[iv][ind_0, :] = 0
        clum = abs(mul_X[iv]).sum(0)
        ind_11 = np.array(np.where(clum != 0)).reshape(-1)
        new_X = np.copy(mul_X[iv][:, ind_11])

        mul_X[iv] = torch.Tensor(np.nan_to_num(np.copy(new_X)))
        del new_X, ind_0, ind_1, ind_11, clum

    WE = torch.Tensor(WE)
    mul_X_trian = [xiv[train_index] for xiv in mul_X]
    mul_X_val = [xiv[val_index] for xiv in mul_X]

    WE_val = WE[val_index]

    args.n_input = [xiv.shape[1] for xiv in mul_X]

    yt_label = np.copy(label[train_index])
    yv_label = np.copy(label[val_index])

    yt_label_clean = np.copy(label_clean[train_index])
    yv_label_clean = np.copy(label_clean[val_index])

    train_max_acc, val_max_acc = train_DIC(mul_X_trian, mul_X_val, WE, WE_val, yt_label, yv_label, yt_label_clean,yv_label_clean, device, args)
    return train_max_acc, val_max_acc




def single_dataset_ratio(dataset,noise,threshoud):
    zz = [2024, 2025, 2026, 2027, 2028]
    args.dataset = dataset
    args.data_path = './Datasets/' + args.dataset
    args.noiseRatio = noise
    val_accs = []
    for item in zz:
        set_seed(item)
        train_acc, val_acc  = main()
        val_accs.append(val_acc)
    avg = np.mean(val_accs)
    std = np.std(val_accs)
    print("noiseRatio:{:.2f}  dataset:{}    res:{:.2f}+{:.2f}".format(args.noiseRatio,args.dataset,avg,std))
    res = "{:.2f}+{:.2f}".format(avg, std)
    return res
import datetime

if __name__ == "__main__":
    ##   PIE, Caltech101, UCI, BBC, Leaves Scene LandUse_21
    # single_dataset_ratio("Caltech101",0.5, 0.5)
    ratios = np.arange(0, 0.55, 0.10)
    # ratios = [0.5]
    # ratios = [0.8]
    # ratios = [0.5]
    # ratios = [0.5]
    datasets = ["ALOI-100","Scene", "LandUse_21" ,"Leaves" , "UCI" , "Caltech101"]
    datasets = ["UCI"]
    args.warmup_type = "New"
    value = []
    for dataset in datasets:
        filename = "./result/final-组合系数随机化/0731{}.txt".format(dataset)
        current_time = "-----{}-----\n".format(datetime.datetime.now())
        ress = []
        if dataset == "Caltech101":
            args.beta = 0.01
        else:
            args.beta = 0.1
        for ratio in ratios:
            if ratio<0.1:
                args.correct=False
            else:
                args.correct = True
            res = single_dataset_ratio(dataset,ratio,0.5)
            value.append(res)
            ress.append(res)






