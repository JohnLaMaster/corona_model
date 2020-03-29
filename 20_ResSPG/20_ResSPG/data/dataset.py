import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
#from config import INPUT_SIZE
from tqdm import tqdm
from scipy.io import loadmat
import random
random.seed(0)
INPUT_SIZE=(448, 448)
def clean_name(name):
    return name
    name = name.replace(' ', '').replace('.jpg', '')
    try:
        int(name)
        return name
    except:
        pass
    while True:
        try:
            int(name[-1])
            name = name[:-1]
        except:
            return name

class SARS():
    def __init__(self, root, is_train=True):
        self.is_train = is_train
        
        path = f'{root}/no_nCoV'
        self.file_lst = [f'{path}/{item}' for item in os.listdir(path)]
        self.people_lst = [clean_name(item.split('_')[0])
                            for item in os.listdir(path)]
        self.label_lst = [0]*len(os.listdir(path))
        
        path = f'{root}/nCoV'
        self.file_lst += [f'{path}/{item}' for item in os.listdir(path)]
        self.people_lst += [clean_name(item.split('_')[0])
                            for item in os.listdir(path)]
        self.label_lst += [1]*len(os.listdir(path))

        temp = list(zip(self.file_lst, self.label_lst, self.people_lst))
        random.shuffle(temp)
        self.file_lst = [item[0] for item in temp]
        self.label_lst = [item[1] for item in temp]
        self.people_lst = [item[2] for item in temp]
        self.file_name = [item.split('/')[-1] for item in self.file_lst]
        
    def __getitem__(self, index):
        flg_H = 0
        img = np.array(Image.open(self.file_lst[index]))
        target = self.label_lst[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img_raw = img.copy()
        img = Image.fromarray(img, mode='RGB')
        img = transforms.Resize((448, 448), Image.BILINEAR)(img)
        if self.is_train:
# =============================================================================
#             img = transforms.RandomCrop(INPUT_SIZE)(img)
# =============================================================================
            if np.random.randint(2) == 1:
                flg_H = 1
                img = transforms.RandomHorizontalFlip(p=1)(img)
            img = transforms.ColorJitter(brightness=0.126, saturation=0.5)(img)
        else:
# =============================================================================
#             img = transforms.CenterCrop(INPUT_SIZE)(img)
# =============================================================================
            pass
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        img_raw = Image.fromarray(img_raw, mode='RGB')
        img_raw = transforms.Resize((600, 600), Image.BILINEAR)(img_raw)
        if flg_H == 1:
            img_raw = transforms.RandomHorizontalFlip(p=1)(img_raw)
        
        img_raw = transforms.ToTensor()(img_raw)
        img_raw = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_raw)

        return img, target, img_raw, self.people_lst[index], self.file_name[index],self.file_lst[index]

    def __len__(self):
        return len(self.label_lst)


