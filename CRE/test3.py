#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import argparse
import os
import torch.nn as nn

from torch.utils.data import DataLoader
from datetime import datetime
from functions import progress_bar
from visualize import Visualizer
from torchnet import meter
import torch.nn.functional as F
from sklearn.metrics import recall_score,precision_score,f1_score,roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from models.ResNet import SupConResNet

from dataset1 import Lung3D_ccii_patient_supcon
# from draw_roc import draw_roc
from tqdm import tqdm


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default=2, type=int,help='model')
parser.add_argument('--pre', '-p', default=0, type=int,help='pre1-nopre0')
parser.add_argument('--batch-size', '-bs', default=32, type=int, help='batch-size')
parser.add_argument('--n_classes', '-n_cls', default=2, type=int, help='n_classes')


def parse_args():
    global args
    args = parser.parse_args()

'''overlap crop: h1n1/h1n1cbam (no mask)'''
@torch.no_grad()
def test():
    parse_args()

    # configure model
    '''pcp'''
    # weight_dir = '/home/houjunlin/2019-nCov/v100-2019-nCov/3DCNN/lung/checkpoints/33dresnet18pre512/239.pkl'
    '''h1n1'''
    weight_dir = '/remote-home/my/2019-nCov/3DCNN/lung/checkpoints/new411/3h1n1res18box256/279.pkl'
    '''h1n1 cbam'''
    # weight_dir = '/remote-home/my/2019-nCov/3DCNN/lung/checkpoints/new411/3h1n1res18box256cbam/229.pkl'
    ckpt = torch.load(weight_dir)
    state_dict = ckpt['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name =k[7:]
        new_state_dict[name] = v
    
    net = resnet18(num_classes=3, spatial_size=256, sample_duration=128)
    # net = resnet18(num_classes=3, spatial_size=256, sample_duration=128, att_type='CBAM')

    net.load_state_dict(new_state_dict)

    net = nn.DataParallel(net)
    net = net.cuda()

    # data
    test_data = Lung3D_np(train=False,inference=True,n_classes=args.n_classes)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    results = []

    net = net.eval()
    correct = .0
    # total = .0
    con_matx = meter.ConfusionMeter(args.n_classes)

    # prob_list = []
    pred_list = []
    label_list = []

    for i, (data,label,id) in enumerate(test_loader):
        print(len(data))
        total = len(data)
        count = .0
        health_rate_list = []
        pcp_max_prob_list = []
        pos_max_prob_list = []
        health_noisy_or_list = []
        pcp_noisy_or_list = []
        pos_noisy_or_list = []
        max_pcp = 0
        max_pos = 0
        pi_health = 1
        pi_pcp = 1
        pi_pos = 1
        for j in range(total):
            cur_data = data[j].unsqueeze(1)
            cur_data = cur_data.float().cuda()
            label = label.float().cuda()
            pred = F.softmax(net(cur_data))
            _, predicted = pred.max(1)

            # prob_list.append(pred.cpu().detach())
            # pred_list.append(predicted.cpu().detach())
            print(pred[0])
            # noisy_or_health
            pi_health = pi_health * (1 - pred[0][0])
            # noisy_or_pcp
            pi_pcp = pi_pcp * (1 - pred[0][1])
            # noisy_or_pos
            pi_pos = pi_pos * (1 - pred[0][2])
            # max pcp prob
            if pred[0][1] >= max_pcp:
                max_pcp = pred[0][1]
            # max pos prob
            if pred[0][2] >= max_pos:
                max_pos = pred[0][2]
            # health prob
            count += pred[0][0]

        health_rate = count/total
        noisy_or_health = 1 - pi_health
        noisy_or_pcp = 1 - pi_pcp
        noisy_or_pos = 1 - pi_pos
        # label_list.append(label.cpu().detach())
        health_rate_list.append(health_rate.cpu().detach())
        pcp_max_prob_list.append(max_pcp.cpu().detach())
        pos_max_prob_list.append(max_pos.cpu().detach())
        health_noisy_or_list.append(noisy_or_health.cpu().detach())
        pcp_noisy_or_list.append(noisy_or_pcp.cpu().detach())
        pos_noisy_or_list.append(noisy_or_pos.cpu().detach())

            # correct += predicted.eq(label.long()).sum().item()        
            # con_matx.add(predicted.detach(),label.detach())

        # predicted = predicted.detach().tolist()
        label = label.detach().tolist()
        # health_rate = health_rate.tolist()
        batch_results = [(id_,label_,health_rate_,pcp_prob_,pos_prob_,pcp_noisyor_,pos_noisyor_,health_noisyor_) for id_,label_,health_rate_,pcp_prob_,pos_prob_,pcp_noisyor_,pos_noisyor_,health_noisyor_ in zip(id,label,health_rate_list,pcp_max_prob_list,pos_max_prob_list,pcp_noisy_or_list,pos_noisy_or_list,health_noisy_or_list) ]
        results += batch_results

    # for i, (data,label,id) in enumerate(test_loader):
    #     data = data.unsqueeze(1)
    #     for j in range(int(data.size(2)//32)):
    #         cur_data = data[:,:,i*32:(i+1)*32,:,:]

    #     data = data.float().cuda()
    #     label = label.float().cuda()
    #     pred = F.softmax(net(data))
    #     _, predicted = pred.max(1)

    #     # prob_list.append(pred.cpu().detach())
    #     pred_list.append(predicted.cpu().detach())
    #     label_list.append(label.cpu().detach())

    #     total += data.size(0)

    #     correct += predicted.eq(label.long()).sum().item()        
    #     con_matx.add(predicted.detach(),label.detach())

    #     predicted = predicted.detach().tolist()
    #     label = label.detach().tolist()
    #     pred = pred.detach().tolist()
    #     batch_results = [(id_,label_,predicted_,pred_) for id_,label_,predicted_,pred_ in zip(id,label,predicted,pred) ]
    #     results += batch_results

    # print(torch.cat(label_list).numpy())
    # print(torch.cat(pred_list).numpy())
    # print(torch.cat(prob_list).numpy())

    # recall = recall_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(), average=None)
    # precision = precision_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
    # print('recall:',recall)
    # print('precision:',precision)


    # f1 = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(), average=None)
    # f1_macro = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(), average='macro')
    # kappa = cohen_kappa_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),weights='quadratic')

    # specificity 
    # for i in range(5):
    #     TN = con_matx.value().sum() - con_matx.value()[i,:].sum() - con_matx.value()[:,i].sum() + con_matx.value()[i,i]
    #     FP = con_matx.value()[:,i].sum() - con_matx.value()[i,i]
    #     specificity = TN/(TN+FP)
    #     print('spe',i,':',specificity)
    # print('recall:',recall)
    # print('f1:',f1)

    # draw_roc(torch.cat(label_list).numpy(), torch.cat(prob_list).numpy())
   
    write_csv(results, result_file)
    # print(correct, total)
    # acc = 100.* correct/total
    # print('acc: ',acc, 'test cm:',str(con_matx.value()))

    return results


'''all: h1n1 / h1h1 cbam (no mask)'''
@torch.no_grad()
def test_all():
    parse_args()

    weight_dir = '/remote-home/my/2019-nCov/CC-CCII/3dlung/checkpoints/con/ccii3d2clfres50supcon/59.pkl'
    # print("4res18nopresupcon3cv1")

    result_file = '/remote-home/my/2019-nCov/2DCNN/lung/results/ccii3d2clfres50supcon.csv'
    print(result_file)
    png_name = 'roc/ccii3d2clfres50supcon'

    ckpt = torch.load(weight_dir)
    state_dict = ckpt['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name =k[7:]
        new_state_dict[name] = v

    net = SupConResNet(name='c2dresnet50', head='mlp', feat_dim=128, n_classes=args.n_classes)
    net.load_state_dict(new_state_dict)

    net = nn.DataParallel(net)
    net = net.cuda()
    net = net.eval()

    # data
    test_data = Lung3D_ccii_patient_supcon(train=False,n_classes=args.n_classes)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    results = []

    correct = .0
    total_sample = .0
    con_matx = meter.ConfusionMeter(args.n_classes)
    prob_list = []
    label_list = []
    pred_list = []

    pbar = tqdm(test_loader, ascii=True)
    for i, (data,label,id) in enumerate(pbar):
        count = .0
        # non_ncp_list = []
        # ncp_list = []

        non_ncp_list = []
        ncp_list = []
        # cp_list = []


        cur_data = data.unsqueeze(1)
        cur_data = cur_data.float().cuda()
        label = label.float().cuda()

        _, pred = net(cur_data)
        pred = F.softmax(pred)
        _, predicted = pred.max(1)

        # print(pred[0])
        p_non_ncp = pred[0][0]
        p_ncp = pred[0][1]
        # p_cp = pred[0][2]

        total_sample += cur_data.size(0)
        correct += predicted.eq(label.long()).sum().item()        
        con_matx.add(predicted.detach(),label.detach())

        non_ncp_list.append(p_non_ncp.cpu().detach().item())
        ncp_list.append(p_ncp.cpu().detach().item())
        # cp_list.append(p_cp.cpu().detach().item())

        # prob_list.append(pred.cpu().detach())
        # label_list.append(label.cpu().detach())
        # pred_list.append(predicted.cpu().detach())

        prob_list.extend(pred.cpu().detach().tolist())
        pred_list.extend(predicted.cpu().detach().tolist())
        label_list.extend(label.cpu().detach().tolist())
        # print(np.array(prob_list).shape)

        # print(pred_list)
        # print(prob_list)
        # print(label_list)

        label = label.detach().tolist()
        # batch_results = [(id_,label_,non_ncp_,ncp_,cp_) for id_,label_,non_ncp_,ncp_,cp_ in zip(id,label,non_ncp_list,ncp_list,cp_list) ]
        batch_results = [(id_,label_,non_ncp_,ncp_) for id_,label_,non_ncp_,ncp_ in zip(id,label,non_ncp_list,ncp_list) ]

        results += batch_results

        pbar.set_description(' acc: %.3f'% (100.* correct / total_sample))
        


    # recall = recall_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(), average=None)
    # precision = precision_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
    recall = recall_score(np.array(label_list), np.array(pred_list), average=None)
    precision = precision_score(np.array(label_list), np.array(pred_list),average=None)   
    print('recall:',recall*100.)
    print('precision:',precision*100.)

    # f1 = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(), average=None)
    # f1_macro = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(), average='macro')
    f1 = f1_score(np.array(label_list), np.array(pred_list), average=None)
    f1_macro = f1_score(np.array(label_list), np.array(pred_list), average='macro')

    # specificity 
    print('sen:',recall*100.)
    for i in range(args.n_classes):
        TN = con_matx.value().sum() - con_matx.value()[i,:].sum() - con_matx.value()[:,i].sum() + con_matx.value()[i,i]
        FP = con_matx.value()[:,i].sum() - con_matx.value()[i,i]
        specificity = TN/(TN+FP)
        print('spec',i,':',specificity*100.)

    # draw_roc(np.array(label_list), np.array(prob_list),png_name,n_classes=args.n_classes)
    # draw_roc(torch.cat(label_list).numpy(), torch.cat(prob_list).numpy(),png_name,n_classes=args.n_classes)
    auc_score = roc_auc_score(np.array(label_list), np.array(prob_list)[:,1])
    print("auc = ", auc_score)

    write_csv(results, result_file)
    print(correct, total_sample)
    acc = 100.* correct/total_sample
    print('acc: ',acc)
    print('test cm:',str(con_matx.value()))

    print('f1:',f1*100.)
    print('f1_macro:',f1_macro*100.)
    return results


def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        # writer.writerow(['id','label','health_rate','pcp_max_prob','pos_max_prob','pcp_noisyor','pos_noisyor','health_noisyor'])
        writer.writerow(['id','label','health','h1n1','pos','cap'])
        writer.writerows(results)


if __name__ == '__main__':
    test_all()
        
