import torchvision
import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import cv2
from torchvision.utils import make_grid, save_image
from resnet import resnet18
from resnet_cbam import resnet18_cbam
from resnet_cbam_mask import resnet18_mask
from dataset import *
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils import visualize_cam, Normalize
from utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer
from torch.utils.data import DataLoader

from resnet_big import SupConResNet, LinearClassifier



class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['model_type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 1, *(input_size), device=device))
                # print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, d, h, w = input.size()
        # print('input size: ', b, c, d, h, w)
        _, logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()
        
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, t, u, v = gradients.size()
        # print('gradient size: ', b, k, t, u, v)

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        # print('saliency map size: ', saliency_map.shape)
        saliency_map = F.upsample(saliency_map, size=(d, h, w), mode='trilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

class GradCAM_mask(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['model_type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            # print('forward activation value: ', output.shape)
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)

        for m in target_layer:
            m.register_forward_hook(forward_hook)
            m.register_backward_hook(backward_hook)
        
        #target_layer.register_forward_hook(forward_hook)
        #target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 1, *(input_size), device=device),torch.zeros(1, 1, *(input_size), device=device))
                # print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, input, mask, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, d, h, w = input.size()
        # print('input size: ', b, c, d, h, w)

        logit = self.model_arch(input, mask)
        
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()
        
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, t, u, v = gradients.size()
        # print('gradient size: ', b, k, t, u, v)

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        # print('saliency map size: ', saliency_map.shape)
        saliency_map = F.upsample(saliency_map, size=(d, h, w), mode='trilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, mask, class_idx=None, retain_graph=False):
        return self.forward(input, mask, class_idx, retain_graph)


class GradCAMpp(GradCAM):
    """Calculate GradCAM++ salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcampp
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcampp = GradCAMpp(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        super(GradCAMpp, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze() 
            
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'] # dS/dA
        activations = self.activations['value'] # A
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        return saliency_map, logit


def gao_xy_mask():
    dst = Lung3D_np_mask(train=False,inference=True,n_classes=3)
    length = dst.__len__()
    
    # base_model = resnet18(num_classes=3, spatial_size=256, sample_duration=128)
    base_model = resnet18(num_classes=3, spatial_size=256, sample_duration=128,att_type='CBAM')
    # weight_dir = 'pretrained_weights/resnet_18_23dataset.pth'
    # weight_dir = '/remote-home/my/2019-nCov/3DCNN/lung/checkpoints/33dresnet18pre512_h1n1seg/629.pkl'
    #weight_dir = '/home/lqwn/2019-nCov/3DCNN/lung/checkpoints/3dresnet18pre512_0303/629.pkl'
    weight_dir = '/remote-home/my/2019-nCov/3DCNN/lung/checkpoints/new411/3h1n1res18box256cbammask/319.pkl'
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['net']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    base_model.load_state_dict(unParalled_state_dict)

    base_model.eval()
    base_model.cuda()
    #print(base_model)

    cam_dict = dict()

    resnet_model_dict = dict(model_type='resnet', arch=base_model, layer_name='layer4', input_size=(128, 256, 256))
    resnet_gradcam = GradCAM_mask(resnet_model_dict,True)
    # resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
    # cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]
    cam_dict['resnet'] = resnet_gradcam


    for indexr in range(length):
        data_list, mask_list, label, ID = dst.__getitem__(indexr)
        for i in range(len(data_list)):
            data = data_list[i]
            lung_mask = mask_list[i]
            for slicer in range(0, 96, 6):
                # slicer = 10
                
                img_slice = torch.from_numpy(data[slicer]).unsqueeze(0).float()
                print(img_slice.shape)
                exit()

                torch_img = cv2.cvtColor(data[slicer].astype('float32'),  cv2.COLOR_GRAY2RGB)
                torch_img = torch.from_numpy(torch_img.transpose(2, 0, 1)).float()
                normed_torch_img  = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().cuda()
                normed_torch_lung_mask  = torch.from_numpy(lung_mask).unsqueeze(0).unsqueeze(0).float().cuda()
                print(normed_torch_img.shape, label, ID, torch_img.shape)
                images = []
                for gradcam in cam_dict.values():
                    mask, _ = gradcam(normed_torch_img,normed_torch_lung_mask,1) #(,y_c)
                    mask = mask.cpu().squeeze().numpy()
                    heatmap, result = visualize_cam(mask[slicer], img_slice)

                    # images.append(torch.stack([torch_img, heatmap, result], 0))
                    images.append(torch.stack([torch_img, result], 0))
                images = make_grid(torch.cat(images, 0), nrow=5)

                output_dir = '3h1n1res18cbammask_h1n1'
                os.makedirs(output_dir, exist_ok=True)
                output_name = ID + '_' + str(i)+ '_' + str(slicer)
                output_path = os.path.join(output_dir, output_name + '.png')
                save_image(images, output_path)


def gao_xy_mask_all():
    dst = Lung3D_np_mask(train=False,inference=True,n_classes=3)
    length = dst.__len__()

    # weight_dir = '/remote-home/my/2019-nCov/3DCNN/lung/checkpoints/new411/3h1n1res18box256cbammask2/229.pkl'
    # weight_dir = '/remote-home/my/2019-nCov/3DCNN/lung/checkpoints/new411/3h1n1res18box256cbammask33/499.pkl'
    '''h1n1 cbam dist2'''
    weight_dir = '/remote-home/my/2019-nCov/3DCNN/lung/checkpoints/new411/13h1n1res18box256cbamdist2/249.pkl'

    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['net']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]

    # base_model = resnet18(num_classes=3, spatial_size=256, sample_duration=280)
    base_model = resnet18(num_classes=3, spatial_size=256, sample_duration=280,att_type='CBAM')
    base_model.load_state_dict(unParalled_state_dict)

    base_model.eval()
    base_model.cuda()
    
    for indexr in range(length):
        data_list, mask_list, label, ID = dst.__getitem__(indexr)


        for i in range(len(data_list)):
            data = data_list[i]
            # print(data.size()) # 280,256,256
            lung_mask = mask_list[i]
            adp_depth = data.shape[0]
            print(ID,data.shape)

            
            # base_model = resnet18(num_classes=3, spatial_size=256, sample_duration=adp_depth,att_type='CBAM')
            # base_model.load_state_dict(unParalled_state_dict)

            # base_model.eval()
            # base_model.cuda()
            # # print(base_model)

            cam_dict = dict()

            resnet_model_dict = dict(model_type='resnet', arch=base_model, layer_name='layer4', input_size=(adp_depth, 256, 256))
            
            resnet_gradcam = GradCAM_mask(resnet_model_dict,True)
            
            # resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
            # cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]
            
            cam_dict['resnet'] = resnet_gradcam
           

            for slicer in range(0, adp_depth, 6):
                # slicer = 10
       
                img_slice = torch.from_numpy(data[slicer]).unsqueeze(0).float()

                torch_img = cv2.cvtColor(data[slicer].astype('float32'),  cv2.COLOR_GRAY2RGB)
                torch_img = torch.from_numpy(torch_img.transpose(2, 0, 1)).float()
                normed_torch_img  = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().cuda()
                normed_torch_lung_mask  = torch.from_numpy(lung_mask).unsqueeze(0).unsqueeze(0).float().cuda()
               
                # print(normed_torch_img.shape, label, ID, torch_img.shape)
                images = []
                for gradcam in cam_dict.values():
                    # print(normed_torch_img.shape)
                    mask, _ = gradcam(normed_torch_img,normed_torch_lung_mask,2) #(,y_c)
                    mask = mask.cpu().squeeze().numpy()
                    # print('mask shape: ', mask.shape, ' | torch_img shape: ', torch_img.shape)
                    heatmap, result = visualize_cam(mask[slicer], img_slice)
                    # for i in range(mask.shape[0]):
                    #     if i < slicer:
                    #         continue
                    #     heatmap, result = visualize_cam(mask[i], img_slice)
                    #     print('heat map size: ', heatmap.shape, ' result shape: ', result.shape)
                    #     break
                    
                    # images.append(torch.stack([torch_img, heatmap, result], 0))
                    images.append(torch.stack([torch_img, result], 0))
                images = make_grid(torch.cat(images, 0), nrow=5)

                output_dir = 'outputs_256_3h1n1cbamdist2'
                os.makedirs(output_dir, exist_ok=True)
                output_name = str(label) + ID + '_' + str(i)+ '_' + str(slicer)
                output_path = os.path.join(output_dir, output_name + '.png')
                save_image(images, output_path)
                del images
        # break

def gao_xy():
    dst = Lung3D_np(train=False,inference=True,n_classes=3)
    length = dst.__len__()
    
    # base_model = resnet18(num_classes=3, spatial_size=256, sample_duration=128)
    base_model = resnet18(num_classes=3, spatial_size=256, sample_duration=128,att_type='CBAM')
    # weight_dir = 'pretrained_weights/resnet_18_23dataset.pth'
    # weight_dir = '/remote-home/my/2019-nCov/3DCNN/lung/checkpoints/33dresnet18pre512_h1n1seg/629.pkl'
    #weight_dir = '/home/lqwn/2019-nCov/3DCNN/lung/checkpoints/3dresnet18pre512_0303/629.pkl'
    weight_dir = '/remote-home/my/2019-nCov/3DCNN/lung/checkpoints/new411/3h1n1res18box256cbam/229.pkl'
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['net']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    base_model.load_state_dict(unParalled_state_dict)

    base_model.eval()
    base_model.cuda()
    print(base_model)

    cam_dict = dict()

    resnet_model_dict = dict(model_type='resnet', arch=base_model, layer_name='layer4', input_size=(128, 256, 256))
    resnet_gradcam = GradCAM(resnet_model_dict,True)
    # resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
    # cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]
    cam_dict['resnet'] = resnet_gradcam


    for indexr in range(length):
        data_list, label, ID = dst.__getitem__(indexr)
        for i in range(len(data_list)):
            data = data_list[i]
            for slicer in range(0, 96, 6):
                # slicer = 10
                
                img_slice = torch.from_numpy(data[slicer]).unsqueeze(0).float()

                torch_img = cv2.cvtColor(data[slicer].astype('float32'),  cv2.COLOR_GRAY2RGB)
                torch_img = torch.from_numpy(torch_img.transpose(2, 0, 1)).float()
                normed_torch_img  = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().cuda()
                print(normed_torch_img.shape, label, ID, torch_img.shape)
                images = []
                for gradcam in cam_dict.values():
                    # print(normed_torch_img.shape)
                    mask, _ = gradcam(normed_torch_img,1) #(,y_c)
                    mask = mask.cpu().squeeze().numpy()
                    # print('mask shape: ', mask.shape, ' | torch_img shape: ', torch_img.shape)
                    heatmap, result = visualize_cam(mask[slicer], img_slice)
                    # for i in range(mask.shape[0]):
                    #     if i < slicer:
                    #         continue
                    #     heatmap, result = visualize_cam(mask[i], img_slice)
                    #     print('heat map size: ', heatmap.shape, ' result shape: ', result.shape)
                    #     break
                    
                    # images.append(torch.stack([torch_img, heatmap, result], 0))
                    images.append(torch.stack([torch_img, result], 0))
                images = make_grid(torch.cat(images, 0), nrow=5)

                output_dir = 'outputs_256_3h1n1cbam_h1n1'
                os.makedirs(output_dir, exist_ok=True)
                output_name = ID + '_' + str(i)+ '_' + str(slicer)
                output_path = os.path.join(output_dir, output_name + '.png')
                save_image(images, output_path)


def gao_xy_all():
    dst = Lung3D_np(train=False,inference=True,n_classes=3)
    length = dst.__len__()

    for indexr in range(length):
        data_list, label, ID = dst.__getitem__(indexr)

        for i in range(len(data_list)):
            data = data_list[i]
            adp_depth = data.shape[0]
            print(ID,data.shape)

            # base_model = resnet18(num_classes=3, spatial_size=256, sample_duration=128)
            base_model = resnet18(num_classes=3, spatial_size=256, sample_duration=adp_depth,att_type='CBAM')
            # weight_dir = 'pretrained_weights/resnet_18_23dataset.pth'
            # weight_dir = '/remote-home/my/2019-nCov/3DCNN/lung/checkpoints/33dresnet18pre512_h1n1seg/629.pkl'
            #weight_dir = '/home/lqwn/2019-nCov/3DCNN/lung/checkpoints/3dresnet18pre512_0303/629.pkl'
            weight_dir = '/remote-home/my/2019-nCov/3DCNN/lung/checkpoints/new411/3h1n1res18box256cbam/229.pkl'
            checkpoint = torch.load(weight_dir)
            state_dict = checkpoint['net']
            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            base_model.load_state_dict(unParalled_state_dict)

            base_model.eval()
            base_model.cuda()
            # print(base_model)

            cam_dict = dict()

            resnet_model_dict = dict(model_type='resnet', arch=base_model, layer_name='layer4', input_size=(adp_depth, 256, 256))
            resnet_gradcam = GradCAM(resnet_model_dict,True)
            # resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
            # cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]
            cam_dict['resnet'] = resnet_gradcam


            for slicer in range(0, adp_depth, 6):
                # slicer = 10
                
                img_slice = torch.from_numpy(data[slicer]).unsqueeze(0).float()

                torch_img = cv2.cvtColor(data[slicer].astype('float32'),  cv2.COLOR_GRAY2RGB)
                torch_img = torch.from_numpy(torch_img.transpose(2, 0, 1)).float()
                normed_torch_img  = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().cuda()
                # print(normed_torch_img.shape, label, ID, torch_img.shape)
                images = []
                for gradcam in cam_dict.values():
                    # print(normed_torch_img.shape)
                    mask, _ = gradcam(normed_torch_img,2) #(,y_c)
                    mask = mask.cpu().squeeze().numpy()
                    # print('mask shape: ', mask.shape, ' | torch_img shape: ', torch_img.shape)
                    heatmap, result = visualize_cam(mask[slicer], img_slice)
                    # for i in range(mask.shape[0]):
                    #     if i < slicer:
                    #         continue
                    #     heatmap, result = visualize_cam(mask[i], img_slice)
                    #     print('heat map size: ', heatmap.shape, ' result shape: ', result.shape)
                    #     break
                    
                    # images.append(torch.stack([torch_img, heatmap, result], 0))
                    images.append(torch.stack([torch_img, result], 0))
                images = make_grid(torch.cat(images, 0), nrow=5)

                output_dir = 'outputs_256_3h1n1cbam_all_all'
                os.makedirs(output_dir, exist_ok=True)
                output_name = str(label) + ID + '_' + str(i)+ '_' + str(slicer)
                output_path = os.path.join(output_dir, output_name + '.png')
                save_image(images, output_path)

def gao_xy_all4():
    dst = Lung3D_np_supcon(train=False,inference=True,n_classes=4)
    # dst = Lung3D_np(train=False,inference=True,n_classes=4)
    length = dst.__len__()
    # base_model = resnet18_cbam(num_classes=4, att_type='CBAM')
    # base_model = resnet18(num_classes=4)
    base_model = SupConResNet()
    weight_dir0 = '/remote-home/my/2019-nCov/3DCNN/lung/checkpoints/61cap4/'
    # weight_dir1 = '4res18nopre4supcon3cv2'
    weight_dir1 = '4res18supcon3cv2'
    weight_dir = weight_dir0 + weight_dir1 + '/289.pkl'

    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['net']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    base_model.load_state_dict(unParalled_state_dict)

    base_model.eval()
    base_model.cuda()

    cam_dict = dict()

    resnet_model_dict = dict(model_type='resnet', arch=base_model, layer_name='layer4', input_size=(128, 256, 256))
    resnet_gradcam = GradCAM(resnet_model_dict,True)
    # resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
    # cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]
    cam_dict['resnet'] = resnet_gradcam
    
    for indexr in range(length):
        data_array, label, ID = dst.__getitem__(indexr)

        for slicer in range(0, data_array.shape[0], 6):
        # slicer = 64
            data = data_array.numpy()
                    
            img_slice = torch.from_numpy(data[slicer]).unsqueeze(0).float()

            torch_img = cv2.cvtColor(data[slicer].astype('float32'),  cv2.COLOR_GRAY2RGB)
            torch_img = torch.from_numpy(torch_img.transpose(2, 0, 1)).float()
            normed_torch_img  = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().cuda()
                # print(normed_torch_img.shape, label, ID, torch_img.shape)
            images = []
            for gradcam in cam_dict.values():
                    # print(normed_torch_img.shape)
                mask, _ = gradcam(normed_torch_img) #(,y_c)
                mask = mask.cpu().squeeze().numpy()
                    # print('mask shape: ', mask.shape, ' | torch_img shape: ', torch_img.shape)
                heatmap, result = visualize_cam(mask[slicer], img_slice)
                    # for i in range(mask.shape[0]):
                    #     if i < slicer:
                    #         continue
                    #     heatmap, result = visualize_cam(mask[i], img_slice)
                    #     print('heat map size: ', heatmap.shape, ' result shape: ', result.shape)
                    #     break
                    
                    # images.append(torch.stack([torch_img, heatmap, result], 0))
                images.append(torch.stack([torch_img, result], 0))
            images = make_grid(torch.cat(images, 0), nrow=5)

            output_dir = 'CAM_xy_' + weight_dir1
            os.makedirs(output_dir, exist_ok=True)
            output_name = str(label) +'_' + ID + '_' + str(slicer)
            output_path = os.path.join(output_dir, output_name + '.png')
            save_image(images, output_path)

def gao_zxy_all4():
    dst = Lung3D_np_supcon(train=False,inference=True,n_classes=4)
    # dst = Lung3D_np(train=False,inference=True,n_classes=4)
    length = dst.__len__()
    # base_model = resnet18_cbam(num_classes=4, att_type='CBAM')
    base_model = resnet18(num_classes=4)
    # base_model = SupConResNet()
    weight_dir0 = '/remote-home/my/2019-nCov/3DCNN/lung/checkpoints/61cap4/'
    weight_dir1 = '4res18nopre4supcon3cv2'
    # weight_dir1 = '4res18supcon3cv2'
    weight_dir = weight_dir0 + weight_dir1 + '/299.pkl'
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['net']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    base_model.load_state_dict(unParalled_state_dict)

    base_model.eval()
    base_model.cuda()
    cam_dict = dict()

    resnet_model_dict = dict(model_type='resnet', arch=base_model, layer_name='layer4', input_size=(128, 256, 256))
    resnet_gradcam = GradCAM(resnet_model_dict,True)
    # resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
    # cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]
    cam_dict['resnet'] = resnet_gradcam

    for indexr in range(length):
        data, label, ID = dst.__getitem__(indexr)

        # for slicer in range(0, data_array.shape[0], 6):
            # slicer = 10
        
        data = data.numpy()
        img_slice = torch.from_numpy(data[:,90,:]).unsqueeze(0).float()

        torch_img = cv2.cvtColor(data[:,90,:].astype('float32'),  cv2.COLOR_GRAY2RGB)
        torch_img = torch.from_numpy(torch_img.transpose(2, 0, 1)).float()
        normed_torch_img  = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().cuda()
        # print(normed_torch_img.shape, label, ID, torch_img.shape)
        images = []
        for gradcam in cam_dict.values():
            # print(normed_torch_img.shape)
            mask, _ = gradcam(normed_torch_img,2) #(,y_c)
            mask = mask.cpu().squeeze().numpy()
            # print('mask shape: ', mask.shape, ' | torch_img shape: ', torch_img.shape)
            heatmap, result = visualize_cam(mask[:,90,:], img_slice)
            # for i in range(mask.shape[0]):
            #     if i < slicer:
            #         continue
            #     heatmap, result = visualize_cam(mask[i], img_slice)
            #     print('heat map size: ', heatmap.shape, ' result shape: ', result.shape)
            #     break
                
            # images.append(torch.stack([torch_img, heatmap, result], 0))
            inv_index = torch.arange(result.size(1) - 1, -1, -1).long()
            result = result[:, inv_index, :]
            torch_img = torch_img[:, inv_index, :]
            
            images.append(torch.stack([torch_img, result], 0))
        images = make_grid(torch.cat(images, 0), nrow=5)

        output_dir = 'CAM_zxy_' + weight_dir1
        os.makedirs(output_dir, exist_ok=True)
        output_name = str(label) +'_' + ID
        print(output_name)
        output_path = os.path.join(output_dir, output_name + '.png')
        save_image(images, output_path)

def gao_zxy():
    dst = Lung3D_np(train=False)
    length = dst.__len__()

    base_model = resnet18(num_classes=2, spatial_size=256, sample_duration=32)
    # weight_dir = 'pretrained_weights/resnet_18_23dataset.pth'
    # weight_dir = '/home/houjunlin/2019-nCov/3DCNN/lung/checkpoints/3dresnetpre/256x256x32/res10/99.pkl'
    weight_dir = '/home/houjunlin/2019-nCov/3DCNN/lung/checkpoints/3dresnetpre/256x256x32doublepre/659.pkl'
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['net']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    base_model.load_state_dict(unParalled_state_dict)

    base_model.eval()
    base_model.cuda()

    cam_dict = dict()

    resnet_model_dict = dict(model_type='resnet', arch=base_model, layer_name='layer4', input_size=(32, 256, 256))
    resnet_gradcam = GradCAM(resnet_model_dict,True)
    # resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
    # cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]
    cam_dict['resnet'] = resnet_gradcam


    for indexr in range(length):
    # for indexr in range(113, 0, -1):
        data, label, ID = dst.__getitem__(indexr)
        slicer = 10

        heatmap_all = []
        result_all = []
        print('data shape: ', data.shape)
        # data = data[::-1,:,:]
        debug_img = cv2.cvtColor(data[::-1,128,:].astype('float32'), cv2.COLOR_GRAY2RGB)
        debug_img = torch.from_numpy(debug_img.transpose(2, 0, 1)).float()
        for piece in range(0, data.shape[0], 32):

            img_slice = torch.from_numpy(data[slicer]).unsqueeze(0).float()
            img_slice_zx = torch.from_numpy(data[piece:piece+32,128,:]).unsqueeze(0).float()

            torch_img = cv2.cvtColor(data[slicer].astype('float32'),  cv2.COLOR_GRAY2RGB)
            torch_img = torch.from_numpy(torch_img.transpose(2, 0, 1)).float()

            # debug_img = cv2.cvtColor(data[piece:piece+32,128,:].astype('float32'), cv2.COLOR_GRAY2RGB)
            # print(debug_img.shape)
            # cv2.imwrite('debug_img.png', debug_img * 255)
            # debug_img = torch.from_numpy(debug_img.transpose(2, 0, 1)).float()

            # break



            # torch_img = torch.from_numpy(torch_img.transpose(2, 0, 1)).float()
            new_data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().cuda()
            print(new_data.shape, label, ID, torch_img.shape)

            images = []
            # changed
            # normed_torch_img = data
            normed_torch_img = new_data[:,:, piece:piece+32, :, :]
            for gradcam in cam_dict.values():
                # print(normed_torch_img.shape)
                # mask, _ = gradcam(normed_torch_img, label)
                mask, _ = gradcam(normed_torch_img)
                mask = mask.cpu().squeeze().numpy()
                # print('mask shape: ', mask.shape, ' | torch_img shape: ', torch_img.shape)
                
                # heatmap, result = visualize_cam(mask[slicer], img_slice)
                # images.append(torch.stack([torch_img, heatmap, result], 0))

                heatmap, result = visualize_cam(mask[:, 128, :], img_slice_zx)
                # print('mask and img_slice_zx and heatmap and result size: ', mask.shape, img_slice_zx.shape, heatmap.shape, result.shape)
                heatmap_all.append(heatmap.cpu())
                result_all.append(result.cpu())

        heatmap_all = torch.cat(heatmap_all, dim=1)    
        result_all = torch.cat(result_all, dim=1)
        inv_index = torch.arange(heatmap_all.size(1) - 1, -1, -1).long()
        heatmap_all = heatmap_all[:, inv_index, :]
        result_all = result_all[:, inv_index, :]
        print('heatmap_all shape: ', debug_img.shape, heatmap_all.shape, result_all.shape)

        images.append(torch.stack([debug_img, heatmap_all, result_all], 0))
        images = make_grid(torch.cat(images, 0), nrow=5)

        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        output_name = ID
        output_path = os.path.join(output_dir, output_name + '.png')
        save_image(images, output_path)
        print('Done i:', indexr, ' ID:', ID)
        print('-' * 50)
        # if indexr > 30:
        #     break


def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    # verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold)
    verts, faces, norm, val = measure.marching_cubes_lewiner(p)


    fig = plt.figure(figsize=(10, 10))
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.savefig('debug_3d.png')
    plt.show()
    

if __name__ == '__main__':
    # gao_xy_mask()
    # gao_xy()
    # gao_xy_all()
    # gao_xy_mask_all()
    gao_zxy_all4()

    # gao_zxy()
    # dst = Lung3D_np(train=False)
    # length = dst.__len__()

    # for indexr in range(length):
    #     if indexr == 0:
    #         continue
    #     data, label, ID = dst.__getitem__(indexr)
    #     print(data.shape, label, ID)
    #     plot_3d(data)
    #     break
