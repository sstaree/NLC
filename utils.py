import sys
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import normalize
import math
import glob, os, shutil
from loss import Loss
import csv
def rank_loss(features1, features2, margin):
    cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) * 2.
    sim = cos(features1, features2)
    diag = torch.diag(sim)
    sim = sim - diag.view(-1,1)
    sim[sim + margin < 0] = 0
    return sim.mean()
def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)

def js_div(p, q):
    # Jensen-Shannon divergence, value is in range (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)



def focal_loss(Y, L, alpha=1, gamma=2):
    """
    计算 Focal Loss
    :param Y: 真实标签, one-hot 编码, 维度为 (n, c)
    :param L: 预测标签, 维度为 (n, c)
    :param alpha: 平衡因子, 默认为 1
    :param gamma: 调节因子, 默认为 2
    :return: Focal Loss 值
    """
    # 确保 L 是概率分布
    L = torch.softmax(L, dim=1)

    # 交叉熵损失
    ce_loss = -Y * torch.log(L + 1e-8)

    # Focal Loss 调节因子
    pt = L * Y
    focal_factor = (1 - pt) ** gamma

    loss = alpha * focal_factor * ce_loss
    return loss.mean()

def NRL(p):
    gamma = 0.01

    NB = p.shape[0]
    p = F.normalize(p, p=2, dim=1)
    d = torch.norm(p[:, None] - p, dim=2, p=2)
    sigma = d.mean()
    r = d ** 2
    w = torch.exp(-r / sigma)
    e = torch.where(gamma - d < 0, 0, gamma - d)
    Lnrl = 1.0 / NB * w.mul(r).sum() + 1.0 / NB * ((1 - w).mul(e ** 2)).sum()
    return Lnrl

def NCT(z):
    t = 0.5
    if z.shape[0]<20:
        return 0
    dist = torch.exp(torch.mm(z, z.t())/t)
    _, neighbors = dist.topk(k=20, dim=1, largest=True, sorted=True)
    loss_cont = 0
    dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = 0

    for i in range(neighbors.shape[0]):
        pos_dist = dist[i][neighbors[i]].sum()
        all_dist = dist[i].sum()
        loss_cont += torch.log(pos_dist/all_dist)
    res = -loss_cont/z.shape[0]
    return 0


def mixup(inputs, targets, alpha):
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)
    idx = torch.randperm(inputs.size(0))
    input_a, input_b = inputs, inputs[idx]
    target_a, target_b = targets, targets[idx]
    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b
    return mixed_input, mixed_target



def eval_train_nce(model,device,mul_X,we,trainNoisyLabels,batch_size,num_class,args,num_neighbor=20):
    model.eval()
    model = model.to(device)
    X = []
    for x in mul_X:
        X.append(x.to(device))
    trainNoisyLabels = torch.tensor(trainNoisyLabels).to(device)
    we = we[0:X[0].shape[0],:]
    we = torch.tensor(we).to(device)
    with torch.no_grad():
        _, trainLogits, trainFeatures, _ = model(X,we)
    trainFeatures = F.normalize(trainFeatures)
    num_batch = math.ceil(float(trainFeatures.size(0)) / batch_size)
    sver_collection = []
    for batch_idx in range(num_batch):
        features = trainFeatures[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        noisy_labels = trainNoisyLabels[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        dist = torch.mm(features, trainFeatures.t())
        dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1  # set self-contrastive samples to -1
        _, neighbors = dist.topk(num_neighbor, dim=1, largest=True, sorted=True)  # find contrastive neighbors


        neighbors = neighbors.view(-1)
        neigh_logits = trainLogits[neighbors]
        neigh_probs = F.softmax(neigh_logits, dim=-1)
        M, _ = features.shape
        given_labels = torch.full(size=(M, num_class), fill_value=0.0001).cuda()
        given_labels.scatter_(dim=1, index=torch.unsqueeze(noisy_labels.long(), dim=1), value=1 - 0.0001)
        given_labels = given_labels.repeat(1, num_neighbor).view(-1, num_class)
        sver = js_div(neigh_probs, given_labels)
        sver_collection += sver.view(-1, num_neighbor).mean(dim=1).detach().cpu().numpy().tolist()
    sver_collection = np.array(sver_collection)
    prob = np.array(sver_collection)

    prob = 1.0 - np.array(sver_collection)
    mask_lab = prob >= args.threshold_sver
    mask_unl = prob < args.threshold_sver

    labeledFeatures = trainFeatures[mask_lab]
    labeledLogits = trainLogits[mask_lab]
    labeledNoisyLabels = trainNoisyLabels[mask_lab]
    labeledX = []
    for item in X:
        labeledX.append(item[mask_lab])
    labeledW = prob[mask_lab]

    unlabeledFeatures = trainFeatures[mask_unl]
    unlabeledLogits = trainLogits[mask_unl]
    unlabeledX = []
    for item in X:
        unlabeledX.append(item[mask_unl])

    knn_labeledLogits = labeledLogits[labeledW > args.knn_threshold]
    knn_labeledFeatures = labeledFeatures[labeledW > args.knn_threshold]
    knn_labeledNoisyLabels = labeledNoisyLabels[labeledW > args.knn_threshold]

    pseudo_noise_ratio =  1 - labeledFeatures.shape[0] / mul_X[0].shape[0]


    num_labeled = knn_labeledFeatures.size(0)
    num_unlabeled = unlabeledFeatures.size(0)
    if num_labeled <= args.num_neighbor:
        pseudo_labels = [-3] * num_unlabeled
        pseudo_labels = np.array(pseudo_labels)
        noisy_labels = trainNoisyLabels.cpu().numpy()
        # print("num_labeled <= args.num_neighbor * 10 ...")
        return prob,noisy_labels,pseudo_noise_ratio

    # caculating pseudo-labels for unlabeled samples
    num_batch_unlabeled = math.ceil(float(unlabeledFeatures.size(0)) / args.batch_size)
    pseudo_labels = []
    scor_collection = []
    for batch_idx in range(num_batch_unlabeled):
        features = unlabeledFeatures[batch_idx * args.batch_size:batch_idx * args.batch_size + args.batch_size]
        logits = unlabeledLogits[batch_idx * args.batch_size:batch_idx * args.batch_size + args.batch_size]
        dist = torch.mm(features, knn_labeledFeatures.t())
        _,neighbors = dist.topk(num_neighbor, dim=1, largest=True, sorted=True)  # find contrastive neighbors
        neighbors = neighbors.view(-1)
        neighs_labels = knn_labeledNoisyLabels[neighbors]
        neighs_logits = knn_labeledLogits[neighbors]
        neigh_probs = F.softmax(neighs_logits, dim=-1)
        neighbor_labels = torch.full(size=neigh_probs.size(), fill_value=0.0001).cuda()
        neighbor_labels.scatter_(dim=1, index=torch.unsqueeze(neighs_labels.long(), dim=1), value=1 - 0.0001)
        scor = js_div(F.softmax(logits.repeat(1, args.num_neighbor).view(-1, num_class), dim=-1), neighbor_labels)
        w = (1 - scor).type(torch.FloatTensor)
        w = w.view(-1, 1).type(torch.FloatTensor).cuda()
        neighbor_labels = (neighbor_labels * w).view(-1, args.num_neighbor, num_class).sum(dim=1)
        pseudo_labels += neighbor_labels.detach().cpu().numpy().tolist()
        scor = scor.view(-1, args.num_neighbor).mean(dim=1)
        scor_collection += scor.detach().cpu().numpy().tolist()
    scor_collection = 1-np.array(scor_collection)
    if len(pseudo_labels)==0:
        noisy_labels = trainNoisyLabels.cpu().numpy()
        return prob, noisy_labels,pseudo_noise_ratio
    pseudo_labels = np.argmax(np.array(pseudo_labels), axis=1)
    noisy_pseudo_labels = trainNoisyLabels.cpu().numpy()[mask_unl]

    noisy_pseudo_labels[scor_collection>args.threshold_scor] = pseudo_labels[scor_collection>args.threshold_scor]
    noisy_labels = trainNoisyLabels.cpu().numpy()
    noisy_labels[mask_unl > 0] = noisy_pseudo_labels

    return prob, noisy_labels,pseudo_noise_ratio







#mixopen
def train(epoch,mul_X,yt_label_clean,yt_label_correct,args,model,device,optimizer,loss_model,loss,WE,index_array):

    num_X = mul_X[0].shape[0]
    # print("clean labels numbers:",num_X)
    train_loss = 0
    train_acc = 0
    index_array = torch.randperm(num_X)
    index_array = [k for k in range(num_X)]
    for batch_idx in range(int(np.ceil(num_X / args.batch_size))):
        idx = index_array[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, num_X)]
        all_input = []
        for iv, X in enumerate(mul_X):
            X = X.detach().cpu()
            all_input.append(X[idx].to(device))

        we = WE[idx].to(device)
        yt_label_correct = torch.tensor(yt_label_correct)

        target_noisy = yt_label_correct[idx].to(device)  # batch_y
        target_clean = torch.tensor(yt_label_clean)[idx].to(device)
        optimizer.zero_grad()


        idx = torch.randperm(all_input[0].shape[0])

        input_b = []
        for iv, X in enumerate(all_input):
            input_b.append(X[idx])

        input_a = all_input
        target_a = target_noisy
        target_b = target_noisy[idx]




        l = np.random.beta(0.5, 0.5)
        l = max(l, 1 - l)
        # print(l)
        if args.mixup == True:
            mixed_target = l * target_a + (1 - l) * target_b
            mixed_input = []
            for iv in range(len(all_input)):
                mixed_input.append(l * input_a[iv] + (1 - l) * input_b[iv])
        else:
            mixed_target = target_a
            mixed_input = input_a


        #original
        x_bar_list, mixed_logits, fusion_z, individual_zs = model(mixed_input, we)


        loss_Cont = 0
        for i in range(len(individual_zs)):
            for j in range(i + 1, len(individual_zs)):
                loss_Cont += rank_loss(individual_zs[i], individual_zs[j],0.5)




        Lx = -torch.mean(torch.sum(F.log_softmax(mixed_logits, dim=1) * mixed_target, dim=1))



        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        Q = torch.softmax(mixed_logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / Q))



        fusion_loss = Lx + args.alpha*penalty +  args.beta * loss_Cont
        # print(args.beta)


        fusion_loss.backward()  # backward
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            x_bar_list, target_pre, fusion_z, individual_zs = model(all_input, we)

        train_acc += np.sum(np.argmax(target_pre.detach().cpu().numpy(), axis=1) == np.argmax(target_clean.detach().cpu().numpy(),axis=1))
        train_loss += fusion_loss.cpu().item()
    return train_loss/num_X,train_acc/num_X



def test(epoch,mul_X_val,yv_label_clean,yt_label_correct,args,model,device,optimizer,loss_model,loss,WE_val,index_array):
    model.eval()
    num_X_val = mul_X_val[0].shape[0]
    val_loss = 0
    val_acc=0
    index_array = torch.randperm(num_X_val)
    num_batch = num_X_val / args.batch_size
    num_batch1 = int(np.ceil(num_X_val / args.batch_size))
    with torch.no_grad():
        for batch_idx in range(int(np.ceil(num_X_val / args.batch_size))):
            idx = index_array[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, num_X_val)]
            val_target_clean = np.array(yv_label_clean)
            val_target_clean = torch.tensor(val_target_clean)
            # print("idx:",idx)
            mul_X_batch = []
            for iv, X in enumerate(mul_X_val):
                mul_X_batch.append(X[idx].to(device))
            we_val = WE_val[idx].to(device)
            sub_target_val = val_target_clean[idx].to(device)  # 相当于batch_y
            x_bar_list_val, target_pre_val, fusion_z_val, individual_zs_val = model(mul_X_batch, we_val)
            val_bat_loss = loss(target_pre_val, sub_target_val, model)
            val_loss += val_bat_loss.cpu().item()
            val_acc += np.sum(np.argmax(target_pre_val.detach().cpu().numpy(), axis=1) == np.argmax(
                sub_target_val.detach().cpu().numpy(), axis=1))


    return val_loss/num_X_val,val_acc/num_X_val


def dense_to_onehot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_onehot = np.zeros((num_labels, num_classes))
    # 展平的索引值对应相加，然后得到精确索引并修改labels_onehot中的每一个值
    labels_onehot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_onehot

