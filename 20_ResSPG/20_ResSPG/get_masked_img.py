import torch
from models import resnet_spg, inception3_spg, resnet_spg_fpn
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from data.dataset import SARS
import os

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
    with torch.no_grad():
        for dat in tqdm(val_loader):
            img, label, people = dat[0], dat[1], dat[2]
            img, label = img.cuda(), label.cuda()
            img_path = dat[5]

            logits = model(img, label)

            last_featmaps = get_localization_maps(logits[-1].data)
            np_last_featmaps = last_featmaps.cpu().data.numpy()
            save_atten = SaveAtten(save_dir=opt.save_dir)
            save_atten.get_masked_img(img_path, np_last_featmaps, label.cpu().numpy())



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