import torch
from models import resnet_spg, inception3_spg, resnet_spg_fpn
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from data.dataset import SARS
import os
import pandas as pd
from config import opt
from utils.get_optimizer import get_finetune_optimizer
from utils.save_atten import SaveAtten, get_localization_maps

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
import warnings

warnings.filterwarnings('ignore')


def train():
    model = resnet_spg_fpn.model(pretrained=True, num_classes=opt.num_classes, threshold=opt.threshold)
    model = model.cuda()
    if opt.model_path:
        model = torch.load(opt.model_path)
    test_path = './input/test/'
    testset = SARS(root=test_path, is_train=False)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,
                                             shuffle=False, num_workers=8, drop_last=False)
    model.eval()
    pred_lst = []
    label_lst = []
    people_lst = []
    name_lst=[]

    with torch.no_grad():
        for dat in tqdm(val_loader):
            img, label, people,names = dat[0], dat[1], dat[2],dat[3]

            img, label = img.cuda(), label.cuda()

            logits = model(img, label)
            logits_1 = F.softmax(logits[0], dim=1)[:, 1]

            pred = [item for item in logits_1.detach().cpu().numpy()]
            label1 = [item for item in label.cpu().numpy()]

            pred_lst += pred
            label_lst += label1
            people_lst += list(people)
            name_lst += list(names)

        auc = roc_auc_score(label_lst, pred_lst)
        print("val_auc", auc)

        pred_dir = {}
        res_dir = {}
        r_dir = {}
        label_dir = {}
        for name in name_lst:
            pred_dir[name] = []
        for a in range(len(name_lst)):
            label_dir[name_lst[a]] = label_lst[a]
        for a in range(len(name_lst)):
            pred_dir[name_lst[a]].append(pred_lst[a])
        for name in pred_dir.keys():
            res_dir[name] = np.mean(pred_dir[name])
        for name in pred_dir.keys():
            if res_dir[name] < 0.785:
                r_dir[name] = 0
            else:
                r_dir[name] = 1
        a_ = 0.0
        b_ = 0.0
        c_ = 0.0
        d_ = 0.0
        for name in pred_dir.keys():
            if label_dir[name] == 1 and r_dir[name] == 1:
                a_ += 1
            elif label_dir[name] == 0 and r_dir[name] == 0:
                d_ += 1
            elif label_dir[name] == 0 and r_dir[name] == 1:
                b_ += 1
            else:
                c_ += 1
        print('person-level_acc', (a_ + d_) / (a_ + b_ + c_ + d_))
        print('person-level specificity', d_ / (b_ + d_))
        print('person-level sensitivity', a_ / (a_ + c_))
        print('person-level precision', (a_) / (a_ + b_))

def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    train()