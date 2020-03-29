# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 17:14:45 2020

@author: SY
"""

import os
import shutil
import numpy as np
'''需要修改的部分'''
input_dir = ''
save_dir = ''
people_lst = os.listdir(input_dir)

def find_all(path):
    result = []
    if path.endswith('.dcm') or path.endswith('.jpg'):
        return [path]
    else:
        for fn in os.listdir(path):
            result += find_all(f'{path}/{fn}')
        return result

def choose_img(img_list_len, choose_nums=15):
    if img_list_len <= 0:
        return []
    else:
        if img_list_len < 10:
            return np.arange(0, img_list_len)
        elif int(0.5 * img_list_len) < choose_nums:
            choose_nums = int(0.5 * img_list_len)
        return np.arange(int(0.25 * img_list_len), int(0.75 * img_list_len), int(0.5 * img_list_len / choose_nums))


for people in people_lst:
    os.makedirs(f'{save_dir}/{people}', exist_ok=True)
    fp_lst = np.array(sorted(find_all(f'{input_dir}/{people}')))
    lst = choose_img(len(fp_lst))
    if len(lst) == 0:
        print(people)
        continue
    fp_lst = fp_lst[lst]
    for fp in fp_lst:
        fn = fp.split('/')[-1]
        shutil.copy(fp, f'{save_dir}/{people}/{people}_{fn}')