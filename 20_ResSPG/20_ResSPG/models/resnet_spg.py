import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import os
import cv2
import numpy as np
# from config import opt
from collections import OrderedDict

import math

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def model(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = resnet50(**kwargs)
        ckpt = model_zoo.load_url(model_urls['resnet50'])
        for module_name in list(model._modules):
            if module_name == 'fc':
                continue
            tmp_dic = OrderedDict({item.replace(module_name+'.', ''):
                ckpt[item] for item in list(ckpt.keys()) 
                if item.startswith(module_name)})
            if len(tmp_dic) == 0:
                continue
            model._modules[module_name].load_state_dict(tmp_dic)
        # model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        return model

    return resnet50(**kwargs)

class SPP_A(nn.Module):
    def __init__(self, in_channels, rates = [1,3,6]):
        super(SPP_A, self).__init__()
        self.aspp = []
        for r in rates:
            self.aspp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=128, kernel_size=3, dilation=r, padding=r),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, out_channels=128, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
            )
        self.out_conv1x1 = nn.Conv2d(128*len(rates), 1, kernel_size=1)

    def forward(self, x):
        aspp_out = torch.cat([classifier(x) for classifier in self.aspp], dim=1)
        return self.out_conv1x1(aspp_out)


class SPP_B(nn.Module):
    def __init__(self, in_channels, num_classes=1000, rates = [1,3,6]):
        super(SPP_B, self).__init__()
        self.aspp = []
        for r in rates:
            self.aspp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=1024, kernel_size=3, dilation=r, padding=r),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(1024, out_channels=1024, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
            )
        self.out_conv1x1 = nn.Conv2d(1024, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        aspp_out = torch.mean(torch.tensor([classifier(x) for classifier in self.aspp]), dim=1)
        return self.out_conv1x1(aspp_out)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, threshold=0.7, 
                 transform_input=True):
        self.inplanes = 64
        self.transform_input = transform_input
        super(ResNet, self).__init__()
        # main layer of ResNet
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        # extra layer
        
        #------------------------------------------
        # inference
        self.fc6 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, dilation=1),   #fc6
            nn.ReLU(True),
        )
        self.th = threshold
        self.fc7_1 = self.apc(1024, num_classes, kernel=3, rate=1)
        self.classier_1 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)  #fc8
        #------------------------------------------
        
        #------------------------------------------
        # Branch B
        self.branchB = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 1, kernel_size=1)
        )
        #------------------------------------------

        #------------------------------------------
        #Segmentation
        self.side3 = self.side_cls(256, kernel_size=3, padding=1)
        self.side4 = self.side_cls(1024, kernel_size=3, padding=1)

        self.side_all = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1, padding=0, dilation=1),
        )

        self.interp = nn.Upsample(size=(224,224), mode='bilinear')
        self._initialize_weights()
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        # self.loss_func = nn.CrossEntropyLoss(ignore_index=255)
        self.loss_func = nn.BCEWithLogitsLoss()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def side_cls(self, in_planes, kernel_size=3, padding=1 ):
        return nn.Sequential(
            nn.Conv2d(in_planes, 512, kernel_size=kernel_size, padding=padding, dilation=1),
            nn.ReLU(inplace=True),
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, label=None):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) # 256
        
        side3 = self.side3(x)
        side3 = self.side_all(side3)
        
        x = self.layer2(x) # 512
        
        x = self.layer3(x) # 1024
        
        side4 = self.side4(x)
        side4 = self.side_all(side4)
        
        x = self.layer4(x) # 2048 feature map
        
        #Branch 1
        out1, last_feat = self.inference(x)
        self.map1 = out1
        
        atten_map = self.get_atten_map(self.interp(out1), label, True)
        
# =============================================================================
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = nn.Dropout(p=0.5)(x)
#         x = self.fc(x)
# =============================================================================
    
        #Branch 2
        out_seg = self.branchB(last_feat)

        logits_1 = torch.mean(torch.mean(out1, dim=2), dim=2)

        return [logits_1, side3, side4, out_seg, atten_map, out1]
    
    def inference(self, x):
        x = F.dropout(x, 0.5)
        x = self.fc6(x)
        x = F.dropout(x, 0.5)
        x = self.fc7_1(x)
        x = F.dropout(x, 0.5)
        out1 = self.classier_1(x)
        return out1, x
    
    def apc(self, in_planes=1024, out_planes=1024, kernel=3, rate=1):
        return nn.Sequential(
            nn.Conv2d(in_planes, 1024, kernel_size=kernel, padding=rate, dilation=rate),   #fc6
            nn.ReLU(True),
            # nn.Conv2d(1024, out_planes, kernel_size=3, padding=1)  #fc8
        )

    def mark_obj(self, label_img, heatmap, label, threshold=0.5):

        if isinstance(label, (float, int)):
            np_label = label
        else:
            np_label = label.cpu().data.numpy().tolist()

        for i in range(heatmap.size()[0]):
            mask_pos = heatmap[i] > threshold
            if torch.sum(mask_pos.float()).data.cpu().numpy() < 30:
                threshold = torch.max(heatmap[i]) * 0.7
                mask_pos = heatmap[i] > threshold
            label_i = label_img[i]
            if isinstance(label, (float, int)):
                use_label = np_label
            else:
                use_label = np_label[i]
            # label_i.masked_fill_(mask_pos.data, use_label)
            label_i[mask_pos.data] = use_label
            label_img[i] = label_i

        return label_img

    def mark_bg(self, label_img, heatmap, threshold=0.1):
        mask_pos = heatmap < threshold
        # label_img.masked_fill_(mask_pos.data, 0.0)
        label_img[mask_pos.data] = 0.0

        return label_img

    def get_mask(self, mask, atten_map, th_high=0.7, th_low = 0.05):
        #mask label for segmentation
        mask = self.mark_obj(mask, atten_map, 1.0, th_high)
        mask = self.mark_bg(mask, atten_map, th_low)

        return mask

    def get_loss(self, logits, gt_labels):
        logits_1, side3, side4, out_seg, atten_map, map1 = logits

        loss_cls = self.loss_cross_entropy(logits_1, gt_labels.long())

        # atten_map = logits[-1]
        mask = torch.zeros((logits_1.size()[0], 224, 224)).fill_(255).cuda()
        mask = self.get_mask(mask, atten_map)

        mask_side4 = torch.zeros((logits_1.size()[0], 224, 224)).fill_(255).cuda()
        mask_side4 = self.get_mask(mask_side4, torch.squeeze(torch.sigmoid(self.interp(side4))), 0.5, 0.05)

        loss_side4 = self.loss_saliency(self.loss_func, self.interp(side4).squeeze(dim=1), mask)
        loss_side3 = self.loss_saliency(self.loss_func, self.interp(side3).squeeze(dim=1), mask_side4)

        fused_atten = (torch.sigmoid(self.interp(side3)) + torch.sigmoid(self.interp(side4)))/2.0

        back_mask = torch.zeros((logits_1.size()[0], 224, 224)).fill_(255).cuda()
        back_mask = self.get_mask(back_mask, torch.squeeze(fused_atten.detach()), 0.7, 0.1)
        loss_back = self.loss_saliency(self.loss_func, self.interp(out_seg).squeeze(dim=1), back_mask)

        loss_val = loss_cls + loss_side3 + loss_side4 + loss_back
        return [loss_val, ]

    def loss_saliency(self, loss_func, logtis, labels):
        positions = labels.view(-1, 1) < 255.0
        return loss_func(logtis.view(-1, 1)[positions], labels.view(-1, 1)[positions])

    def loss_segmentation(self, loss_func, logits, labels):
        logits = logits.permute(0, 2, 3, 1).contiguous().view((-1, self.num_classes+1))
        labels = labels.view(-1).long()

        return loss_func(logits, labels)

    def loss_erase_step(self, logits, gt_labels):
        loss_val = 0
        # for single label images
        logits = F.softmax(logits, dim=1)

        if len(logits.size()) != len(gt_labels.size()):

            for batch_idx in range(logits.size()[0]):
                loss_val += logits[batch_idx, gt_labels.cpu().data.numpy()[batch_idx]]

        #try torch.sum(logits[:,gt_labels])

        return loss_val/(logits.size()[0])

    def save_erased_img(self, img_path, img_batch=None):
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
        if img_batch is None:
            img_batch = self.img_erased
        if len(img_batch.size()) == 4:
            batch_size = img_batch.size()[0]
            for batch_idx in range(batch_size):
                imgname = img_path[batch_idx]
                nameid = imgname.strip().split('/')[-1].strip().split('.')[0]

                # atten_map = F.upsample(self.attention.unsqueeze(dim=1), (321,321), mode='bilinear')
                atten_map = F.upsample(self.attention.unsqueeze(dim=1), (224,224), mode='bilinear')
                # atten_map = F.upsample(self.attention, (224,224), mode='bilinear')
                # mask = torch.sigmoid(20*(atten_map-0.5))
                mask = atten_map
                mask = mask.squeeze().cpu().data.numpy()

                img_dat = img_batch[batch_idx]
                img_dat = img_dat.cpu().data.numpy().transpose((1,2,0))
                img_dat = (img_dat*std_vals + mean_vals)*255

                img_dat = self.add_heatmap2img(img_dat, mask)
                save_path = os.path.join('../save_bins/', nameid+'.png')
                cv2.imwrite(save_path, img_dat)
                # save_path = os.path.join('../save_bins/', nameid+'..png')
                # cv2.imwrite(save_path, mask*255)

    def add_heatmap2img(self, img, heatmap):
        # assert np.shape(img)[:3] == np.shape(heatmap)[:3]

        heatmap = heatmap* 255
        color_map = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        img_res = cv2.addWeighted(img.astype(np.uint8), 0.5, color_map.astype(np.uint8), 0.5, 0)

        return img_res

    def get_localization_maps(self):
        map1 = self.normalize_atten_maps(self.map1)
        # map_erase = self.normalize_atten_maps(self.map_erase)
        # return torch.max(map1, map_erase)
        return map1

    def get_feature_maps(self):
        return self.normalize_atten_maps(self.map1)

    def get_heatmaps(self, gt_label):
        map1 = self.get_atten_map(self.map1, gt_label)
        return [map1,]

    def get_fused_heatmap(self, gt_label):
        maps = self.get_heatmaps(gt_label=gt_label)
        fuse_atten = maps[0]
        return fuse_atten

    def get_maps(self, gt_label):
        map1 = self.get_atten_map(self.map1, gt_label)
        return [map1, ]

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def get_atten_map(self, feature_maps, gt_labels, normalize=True):
        label = gt_labels.long()

        feature_map_size = feature_maps.size()
        batch_size = feature_map_size[0]

        atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]])
        atten_map = atten_map.cuda()
        for batch_idx in range(batch_size):
            atten_map[batch_idx, : ,:] = torch.squeeze(feature_maps[batch_idx, label.data[batch_idx], :,:])

        if normalize:
            atten_map = self.normalize_atten_maps(atten_map)

        return atten_map


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
