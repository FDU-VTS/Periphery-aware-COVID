#!/usr/bin/env python
# coding: utf-8
import cv2
import os
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import sys
from utils import *
from tqdm import tqdm
from dataset1 import CCII_distmap
from torch.utils.data import DataLoader
import torch.nn.functional as F
from visualize import Visualizer
from torchnet import meter
# from functions import progress_bar
from datetime import datetime
from sklearn.metrics import precision_score, recall_score
from resnet1 import Covidnet
from utils import compute_pixel_level_metrics, batch_intersection_union, batch_pix_accuracy, diceloss
from utils import GDiceLoss
from torchvision.utils import make_grid,save_image
import torch.backends.cudnn as cudnn
import random

print("torch = {}".format(torch.__version__))

IMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--visname', '-vis', default='2ddistpre', help='visname')
parser.add_argument('--batch_size', '-bs', default=32, type=int, help='batch_size')
parser.add_argument('--lr', '-lr', default=1e-4, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n_cls', default=6, type=int, help='n_classes')
parser.add_argument('--pre', '-pre', default=False, type=bool, help='use pretrained')
parser.add_argument('--mask', '-mask', default=False, type=bool, help='if mask')

best_acc = 0
best_iou = 0
val_epoch = 1
save_epoch = 5

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.deterministic = True

def parse_args():
    global args
    args = parser.parse_args()

def get_lr(cur, epochs):
    if cur < int(epochs * 0.3):
        lr = args.lr
    elif cur < int(epochs * 0.8):
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    return lr

# prepare your own data
def main():
    global best_acc
    global save_dir

    parse_args()
    vis = Visualizer(args.visname,port=9000)

    target_model = Covidnet(pretrained=True)
    target_model = nn.DataParallel(target_model)
    target_model = target_model.cuda()

    # prepare data  
    train_data = CCII_distmap(train=True)
    val_data = CCII_distmap(train=False)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(target_model.parameters(), args.lr, weight_decay=1e-5)
    con_matx = meter.ConfusionMeter(args.n_classes)

    save_dir = './checkpoints/' + args.visname
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  

    # train the model
    initial_epoch = 0
    for epoch in range(initial_epoch, initial_epoch + args.epochs):
        target_model.train()
        con_matx.reset()
        total_loss = .0
        total = .0
        correct = .0
        count = .0
        total_num = .0
        total_acc = .0
        total_iou = .0

        lr = get_lr(epoch, args.epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        pred_list = []
        label_list = []
        
        pbar = tqdm(train_loader, ascii=True)
        for i, (data, label, ID) in enumerate(pbar):               
            data = data.float().cuda()
            label = label.float().cuda()

            pred = target_model(data)
            _, predicted = pred.max(1)
            loss = criterion(pred, label.long())

            # save_fig(0,data,predicted,label,ID,1,'pred')

            acc = batch_pix_accuracy(pred, label.long())
            iou = batch_intersection_union(pred, label.long())

            total_loss += loss.item()
            total_acc += acc
            total_iou += iou
            count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            pbar.set_description('loss:%.3f' % (total_loss / (i+1)) + ' acc: %.3f'% (total_acc / (i+1)) + ' iou: %.3f'% (total_iou / (i+1)))

        vis.plot('loss', total_loss/count)
        vis.log('epoch:{epoch},lr:{lr},loss:{loss},acc:{acc},iou:{iou}'.format(epoch=epoch,lr=lr,loss=total_loss/count,acc=total_acc/count,iou=total_iou/count))
            
        if (epoch + 1) % val_epoch == 0:
            val(target_model,val_loader,epoch,vis)

@torch.no_grad()
def val(net, val_loader, epoch, vis):
    global best_acc
    global best_iou
    parse_args()
    net = net.eval()

    correct = .0
    total = .0
    pred_list = []
    label_list = []
    total_acc = .0
    total_iou = .0

    pbar = tqdm(val_loader, ascii=True)
    for i, (data,label,ID) in enumerate(pbar):
        data = data.float().cuda()
        label = label.float().cuda()
        pred = net(data)
        _, predicted = pred.max(1)

        if (epoch+1) % 10 == 0:
            save_fig(epoch,data,predicted,label,ID,0,'pred')

        acc = batch_pix_accuracy(pred, label.long())
        iou = batch_intersection_union(pred, label.long())

        total_acc += acc
        total_iou += iou
        total += 1

        pbar.set_description(' acc: %.3f'% (total_acc / (i+1)) + ' iou: %.3f'% (total_iou / (i+1)))

    print('val epoch:', epoch, ' val acc: ', total_acc / total, ' val iou:', total_iou / total)
    vis.plot('val acc', total_acc/total)
    vis.plot('val iou', total_iou/total)
    vis.log('epoch:{epoch},val_acc:{val_acc},val_iou:{val_iou}'.format(epoch=epoch,val_acc=total_acc/total,val_iou=total_iou/total))   

    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        }
    save_name = os.path.join(save_dir, str(epoch) + '.pkl')
    torch.save(state, save_name)
    best_iou = total_iou / total

def save_fig(epoch, data, pred, label, ID, train_mode, mode='pred'):
    if train_mode:
        dir = 'imgs/train/' + str(epoch) +'/'
    else:
        dir = 'imgs/test/' + str(epoch) +'/'
    os.makedirs(dir, exist_ok=True)

    for bs in range(20):
        img_data = data[bs]
        img_pred = pred[bs]
        img_label = label[bs]
        name = str(ID[bs])

        torch_img = img_data.cpu()
        torch_label1 = cv2.cvtColor(img_label.cpu().numpy().astype('float32'),  cv2.COLOR_GRAY2RGB)
        torch_pred1 = cv2.cvtColor(img_pred.cpu().numpy().astype('float32'),  cv2.COLOR_GRAY2RGB)

        torch_label1 = torch_label1.transpose(2,0,1)
        torch_pred1 = torch_pred1.transpose(2,0,1)

        torch_img = torch_img.float()
        torch_label = img_label.float()
        torch_pred = img_pred.float()

        torch_label1 = torch.from_numpy(torch_label1).float()
        torch_pred1 = torch.from_numpy(torch_pred1).float()

        color_label, color_pred = drawcolor(torch_label,torch_pred)
        imgs = torch.stack([torch_img,torch_label1*255,torch_pred1*255,\
        color_label,color_pred], 0)

        imgs = make_grid(imgs, nrow=5)
        save_image(imgs, dir + name + mode + '.png')

def drawcolor(torch_label,torch_pred):
    # print(np.unique(np.array(torch_pred.cpu())))
    import torch
    color_label = torch.zeros(3,256,256)
    color_pred = torch.zeros(3,256,256)
    color = [color_label, color_pred]
    torch = [torch_label, torch_pred]

    for i in range(2):
        cur_torch = torch[i]
        cur_color = color[i]

        area = cur_torch==0
        cur_color[0,area]=0
        cur_color[1,area]=0
        cur_color[2,area]=0

        area = cur_torch==1
        cur_color[0,area]=0/255
        cur_color[1,area]=0/255
        cur_color[2,area]=255/255

        area = cur_torch==2
        cur_color[0,area]=30/255
        cur_color[1,area]=144/255
        cur_color[2,area]=255/255

        area = cur_torch==3
        cur_color[0,area]=102/255
        cur_color[1,area]=205/255
        cur_color[2,area]=0

        area = cur_torch==4
        cur_color[0,area]=238/255
        cur_color[1,area]=180/255
        cur_color[2,area]=34/255

        area = cur_torch==5
        cur_color[0,area]=255/255
        cur_color[1,area]=255/255
        cur_color[2,area]=0
    return color_label, color_pred


if __name__ == "__main__":
    main()
