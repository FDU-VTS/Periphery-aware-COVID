import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch import einsum

def diceloss(pred, target):
    batch = pred.size(0)
    
    print(pred.shape, target.shape)
    pred = pred.view(batch, -1)
    target = target.view(batch, -1)
    smooth = 1

    intersection = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)

    total =  2.0 * (intersection + smooth) / (union + smooth)
    return 1 - total.mean()




class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def get_average(self):
        return self.avg
    
    def get_sum(self):
        return self.sum

def compute_pixel_level_metrics(pred, target):
    """ Compute the pixel-level tp, fp, tn, fn between
    predicted img and groundtruth target
    """

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    tp = np.sum(pred * target)  # true postives
    tn = np.sum((1-pred) * (1-target))  # true negatives
    fp = np.sum(pred * (1-target))  # false postives
    fn = np.sum((1-pred) * target)  # false negatives

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    F1 = 2 * precision * recall / (precision + recall + 1e-10)
    acc = (tp + tn) / (tp + fp + tn + fn + 1e-10)
    performance = (recall + tn/(tn+fp+1e-10)) / 2
    iou = tp / (tp+fp+fn+1e-10)

    return [acc, iou, recall, precision, F1, performance]



def batch_intersection_union(output, target, nclass=2):
    prediction = torch.max(output, 1)[1]

    prediction = prediction.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    # prediction = prediction * (target > 0).astype(prediction.dtype)
    intersection = prediction * (prediction == target)

    range_min = 1
    range_max = nclass

    area_inter, _ = np.histogram(intersection, bins = nclass, range=(range_min, range_max))
    area_pred, _ = np.histogram(prediction, bins = nclass, range=(range_min, range_max))
    area_label, _ = np.histogram(target, bins = nclass, range=(range_min, range_max))
    
    area_union = area_label + area_pred - area_inter
    assert (area_inter <= area_union).all(),  "Intersection area should be smaller than Union area"
    
    # return area_inter, area_union
    
    # print(area_inter * 1.0 / area_union).mean()
    return (area_inter * 1.0 / area_union).mean()

def batch_pix_accuracy(output, target, nclass=2):
    prediction = torch.max(output, 1)[1]

    prediction = prediction.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    prediction = prediction * (target > 0).astype(prediction.dtype)
    correct = np.sum(prediction == target)
    labeled = np.sum(np.ones_like(target))
    # labeled = np.sum(target > 0)

    assert (correct <= labeled).all(), "Correct area should be smaller than Labeled"
    # return correct, labeled
    
    # print(correct * 1.0 / labeled)
    return correct * 1.0 / labeled








class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        
        softmax_output = self.apply_nonlin(net_output) if self.apply_nonlin else net_output
    
        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = 1 / (einsum("bcxyz->bc", y_onehot).type(torch.float32) + 1e-10)**2
        intersection: torch.Tensor = w * einsum("bcxyz, bcxyz->bc", softmax_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxyz->bc", softmax_output) + einsum("bcxyz->bc", y_onehot))
        divided: torch.Tensor = 1 - 2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc
