import numpy as np
import cv2
import os
import torch
from utils.anchor import get_bounding_box
from config import opt

def adjust_box(box):
    y0, x0, y1, x1 = box
    y0, x0 = max(y0, 0), max(x0, 0)
    y1, x1 = min(y1, 1024), min(x1, 1024)
    return np.array([y0, x0, y1, x1]).astype(np.int32)


def normalize_atten_maps(atten_maps):
    atten_shape = atten_maps.size()
    batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
    batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
    atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,)) - batch_mins,
                             batch_maxs - batch_mins)
    atten_normed = atten_normed.view(atten_shape)

    return atten_normed


def get_localization_maps(map1):
    map1 = normalize_atten_maps(map1)
    return map1


class SaveAtten(object):
    def __init__(self, save_dir='save_bins'):
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_masked_img_batch(self, path_batch, atten_batch, label_batch):
        img_num = atten_batch.shape[0]
        bounding_boxes = []
        print(range(img_num),len(path_batch))
        for idx in range(img_num):
            atten = atten_batch[idx]
            label = label_batch[idx]
            label = int(label)
            bounding_boxes.append(self._save_masked_img(path_batch[idx], atten, label))
        return bounding_boxes


    # save masked images with only on ground truth label
    def _save_masked_img(self, img_path, atten, label):
        if not os.path.isfile(img_path):
            raise('Image not exist')
        img = cv2.imread(img_path)
        w, h = img.shape[:2]
        attention_map = atten[label, :, :]
        atten_norm = attention_map

        img_max = np.max(atten_norm)
        # print("********")
        # print(img_max)
        loc = np.where(img_max == atten_norm)
        # print(len(loc[0]))
        # print(loc)

        # min_val = np.min(attention_map)
        # max_val = np.max(attention_map)
        # atten_norm = (attention_map - min_val)/(max_val - min_val)

        atten_norm = cv2.resize(atten_norm, dsize=(h, w))
        atten_norm = atten_norm * 255
        img = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)
        for y, x in zip(loc[0], loc[1]):
# =============================================================================
#             y = int(y * 25.6) # v3-321-40
#             x = int(x * 25.6)
# =============================================================================
            y = int(y * 73.1) # resnet50-448-14
            x = int(x * 73.1)
            
            bounding_box = get_bounding_box(img, x, y, opt.anchor.copy(), stride=opt.stride)
            bounding_box = adjust_box(bounding_box)
            cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
            y0, x0, y1, x1 = bounding_box
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 5)
        # img = cv2.addWeighted(img.astype(np.uint8), 0.5, heat_map.astype(np.uint8), 0.5, 0)

# =============================================================================
#         img_id = img_path.strip().split('/')[-1]
#         save_dir = os.path.join(self.save_dir, img_id)
#         cv2.imwrite(save_dir, img)
# =============================================================================
        return bounding_box

    def get_heatmap_idxes(self, gt_label):
        labels_idx = []
        if np.ndim(gt_label) == 1:
            labels_idx = np.expand_dims(gt_label, axis=1).astype(np.int)
        elif np.ndim(gt_label) == 2:
            for row in gt_label:
                idxes = np.where(row[0] == 1)[0] if np.ndim(row) == 2 else np.where(row == 1)[0]
                labels_idx.append(idxes.tolist())
        else:
            labels_idx = None

        return labels_idx

    def get_map_k(self, atten, k, size=(224, 224)):
        atten_map_k = atten[k, :, :]
        atten_map_k = cv2.resize(atten_map_k, dsize=size)
        return atten_map_k

    def read_img(self, img_path, size=(224, 224)):
        img = cv2.imread(img_path)
        if img is None:
            print("Image does not exist. %s" %(img_path))
            exit(0)

        if size == (0, 0):
            size = np.shape(img)[:2]
        else:
            img = cv2.resize(img, size)
        return img, size[::-1]

    def normalize_map(self, atten_map):
        min_val = np.min(atten_map)
        max_val = np.max(atten_map)
        atten_norm = (atten_map - min_val)/(max_val - min_val)

        return atten_norm

    def _add_msk2img(self, img, msk, isnorm=True):
        if np.ndim(img) == 3:
            assert np.shape(img)[0:2] == np.shape(msk)
        else:
            assert np.shape(img) == np.shape(msk)

        if isnorm:
            min_val = np.min(msk)
            max_val = np.max(msk)
            atten_norm = (msk - min_val)/(max_val - min_val)
        atten_norm = atten_norm * 255
        heat_map = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)
        w_img = cv2.addWeighted(img.astype(np.uint8), 0.5, heat_map.astype(np.uint8), 0.5, 0)

        return w_img

    def get_masked_img(self, img_path, atten, gt_label, size=(224, 224),
                       maps_in_dir=False, save_dir=None, only_map=False):
        assert np.ndim(atten) == 4

        save_dir = save_dir if save_dir is not None else self.save_dir

        if isinstance(img_path, list) or isinstance(img_path, tuple):
            batch_size = len(img_path)
            label_indexes = self.get_heatmap_idxes(gt_label)
            for i in range(batch_size):
                img, size = self.read_img(img_path[i], size)
                img_name = img_path[i].split('/')[-1]
                img_name = img_name.strip().split('.')[0]
                if maps_in_dir:
                    img_save_dir = os.path.join(save_dir, img_name)
                    os.mkdir(img_save_dir)

                for k in label_indexes[i]:
                    atten_map_k = self.get_map_k(atten[i], k, size)
                    msked_img = self._add_msk2img(img, atten_map_k)

                    suffix = str(k + 1)
                    if only_map:
                        save_img = (self.normalize_map(atten_map_k) * 255).astype(np.int)
                    else:
                        save_img = msked_img
                    print(os.path.join(save_dir, img_name + '_' + suffix + '.png'))
                    if maps_in_dir:
                        cv2.imwrite(os.path.join(img_save_dir, suffix + '.png'), save_img)
                    else:
                        cv2.imwrite(os.path.join(save_dir, img_name + '_' + suffix + '.png'), save_img)

