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
    
    os.makedirs('checkpoints', exist_ok=True)
    model = resnet_spg_fpn.model(pretrained=True, num_classes=opt.num_classes, threshold=opt.threshold)
    model=model.cuda()
    if opt.model_path:
        model = torch.load(opt.model_path)
    model = torch.nn.DataParallel(model)
    optimizer = get_finetune_optimizer(model, opt.lr)

    train_path = './input/train/'
    test_path = './input/test/'
    trainset = SARS(root=train_path, is_train=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                              shuffle=True, num_workers=8, drop_last=False)
    testset = SARS(root=test_path, is_train=False)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,
                                             shuffle=False, num_workers=8, drop_last=False)

    total_epoch = opt.max_epoch

    for t in range(1, total_epoch):
        print('***' * 30)
        print('epoch', t)
        pred_lst = []
        label_lst = []
        
        people_lst = []
        
        model.train()

        for dat in tqdm(train_loader):
            img, label, people = dat[0], dat[1], dat[2]
            img, label = img.cuda(), label.cuda()
            logits = model(img, label)

            logits_1 = F.softmax(logits[0], dim=1)[:, 1]
            loss_val, = model.module.get_loss(logits, label)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            pred = [item for item in logits_1.detach().cpu().numpy()]
            label1 = [item for item in label.cpu().numpy()]

            pred_lst += pred
            label_lst += label1
            people_lst += list(people)
        

        train_auc = roc_auc_score(label_lst, pred_lst)
        print("train_auc", train_auc)
        #np.save('./train_people.npy', np.array(people_lst))
        np.save('./train_people.npy', [np.array(t) for t in people_lst])
        np.save('./train_pred.npy', np.array(pred_lst))
        np.save('./train_label.npy', np.array(label_lst))

        torch.save(model, os.path.join('checkpoints', 'fourth' + str(t).zfill(3) + '.pkl'))
        model.eval()
        
        pred_lst = []
        label_lst = []
        people_lst = []
        with torch.no_grad():
            for dat in tqdm(val_loader):
                img, label, people = dat[0], dat[1], dat[2]
                img, label = img.cuda(), label.cuda()
                
                logits = model(img, label)
                logits_1 = F.softmax(logits[0], dim=1)[:, 1]

                pred = [item for item in logits_1.detach().cpu().numpy()]
                label1 = [item for item in label.cpu().numpy()]

                pred_lst += pred
                label_lst += label1
                
                people_lst += list(people)


        auc = roc_auc_score(label_lst, pred_lst)
        print("val_auc", auc)
            
        #np.save('./test_people.npy', np.array(people_lst))
        np.save('./test_people.npy', [np.array(t) for t in people_lst])
        np.save('./test_pred.npy', np.array(pred_lst))
        np.save('./test_label.npy', np.array(label_lst))

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