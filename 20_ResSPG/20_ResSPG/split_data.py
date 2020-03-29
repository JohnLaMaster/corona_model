import os
import random

random.seed(575)

'''需要修改的部分'''
nCoV = [] #
no_nCoV = []

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


testsize = 0.4

no_nCoV_file_lst = []
nCoV_file_lst = []

for dataset in no_nCoV:
    no_nCoV_path = f'./{dataset}/crop'
    no_nCoV_file_lst += [f'{no_nCoV_path}/{item}' for item in os.listdir(no_nCoV_path)]
for dataset in nCoV:
    nCoV_path = f'./{dataset}/crop'
    nCoV_file_lst += [f'{nCoV_path}/{item}' for item in os.listdir(nCoV_path)]

no_nCoV_people_lst = list(set([clean_name(item.split('/')[-1].split('_')[0])
                               for item in no_nCoV_file_lst]))
nCoV_people_lst = list(set([clean_name(item.split('/')[-1].split('_')[0])
                            for item in nCoV_file_lst]))
print(len(no_nCoV_people_lst))
print(len(nCoV_people_lst))

random.shuffle(no_nCoV_people_lst)
random.shuffle(nCoV_people_lst)

for file_lst, people_lst, name in [[no_nCoV_file_lst, no_nCoV_people_lst, 'no_nCoV'],
                                   [nCoV_file_lst, nCoV_people_lst, 'nCoV']]:

    train_num = int((1 - testsize) * len(people_lst))
    train_people_lst = set(people_lst[:train_num])
    test_people_lst = set(people_lst[train_num:])
    train_lst = [item for item in file_lst
                 if clean_name(item.split('/')[-1].split('_')[0]) in train_people_lst]
    test_lst = [item for item in file_lst
                if clean_name(item.split('/')[-1].split('_')[0]) in test_people_lst]

    import shutil

    os.makedirs('./input/train', exist_ok=True)
    os.makedirs('./input/val', exist_ok=True)
    os.makedirs('./input/test', exist_ok=True)
    os.makedirs(f'./input/train/{name}', exist_ok=True)
    os.makedirs(f'./input/val/{name}', exist_ok=True)
    os.makedirs(f'./input/test/{name}', exist_ok=True)

    for fp in train_lst:
        fn = fp.split('/')[-1]
        shutil.copy(fp.replace('crop', 'crop'),
                    f'./input/train/{name}/{fn}')
        # =============================================================================
        #         shutil.copy(fp,
        #                     f'./input/train/{name}/{fn}'.replace('.jpg', '_filled.jpg'))
        # =============================================================================
        shutil.copy(fp.replace('crop', 'crop'),
                    f'./input/val/{name}/{fn}')
    for fp in test_lst:
        fn = fp.split('/')[-1]
        shutil.copy(fp.replace('crop', 'crop'),
                    f'./input/test/{name}/{fn}')
