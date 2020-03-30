# Version 0.0.2

# Pre-processing data for the chinese team's COVID-19 model
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from auxiliary import *
from PIL import Image


# Primary Workload
class PreprocessFiles():
    def __init__(self, opt):
        super(PreprocessFiles, self).__init__()
        self.opt = opt
        self.path = opt.path
        if opt.savedir:
            self.savedir = opt.savedir 
            if not os.path.exists(self.savedir) or not os.path.isdir(self.savedir):
                os.makedirs(self.savedir)
        else: 
            self.savedir = self.path

    def convert2nifti(self):
        print('>>> Converting dicom files to nifti...')
        self.nifti = convertdcm(self.opt.path, self.opt)

    def segment(self):
        print('>>> Segmenting nifti files...')
        # seg = segment(paths) # Here is for the segmentation

    def makeDataset(self):
        print('>>> Compiling dataset...')
        if self.opt.d2n:
            self.paths = zip(sorted(self.nifti), sorted(self.seg))
        else:
            self.paths = make_dataset(self.opt.path, self.opt)

    def ct2jpg(self):
        print('>>> Processing CT scans and converting to jpg...')
        for path1, path2 in self.paths:
            file1, file2 = nib.load(path1), nib.load(path2)
            header = file1.header
            medical_image = np.asarray(file1.get_fdata())
            segmentation = np.asarray(file2.get_fdata())

            # Hu windowing
            segmentation = resample(segmentation,header['pixdim'][3],header['pixdim'][1:3])
            image = resample(medical_image,header['pixdim'][3],header['pixdim'][1:3])
            image = transform_to_hu(header, image)
            image, mn, mx = window_image(image, self.opt.window_center, self.opt.window_width)

            # Slice selection
            mask_quant = segmentation.sum((0,1))
            mask_quant /= np.max(mask_quant)
            tInd = np.squeeze(np.where(mask_quant>=self.opt.threshold))
            ind = np.arange(start=tInd[0], stop=tInd[-1], step=self.opt.slice)
            image = image[:,:,ind]
            image = normalize(image, mn, mx, 0, 255)

            # Exporting each slice as a jpeg
            parent, base = os.path.split(path1)
            root, ext = os.path.splitext(base)
            path = os.path.join(self.savedir, root)
            if not os.path.exists(path): os.makedirs(path)
            for i in range(image.shape[2]):
                pathname = root + '_slice_{}'.format(i) + '.jpg'
                pathname = os.path.join(path, pathname)

                im = Image.fromarray(image[:,:,i])
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                im.save(pathname)

            if not self.opt.no_plotDist:
                self.plot_dist(mask_quant, ind, path, root)
                
    def plot_dist(self, mask_quant, ind, path, name):
        name += '_segmentation_distribution_&_selection'
        path = os.path.join(path,name)
        x = np.asarray(range(len(mask_quant)))
        y = np.ones(x.size) * self.opt.threshold
        nel = np.argwhere(mask_quant>=self.opt.threshold)

        plt.figure(figsize=(15, 10))
        plt.bar(range(len(mask_quant)),mask_quant, zorder=1)
        plt.plot(ind,np.zeros(ind.size),'|r',zorder=2,label='Selected Slices')
        plt.plot(ind,np.zeros(ind.size),'|r',markersize=72)
        plt.plot(x,y,'r-',label='Threshold')
        plt.ylim(0)
        plt.title('Slice Distribution')
        plt.xlabel('Number of slices over threshold, {}: {}'.format(self.opt.threshold,len(nel)))
        plt.legend(loc=1)
        
        # plt.savefig(path+'.png',format='png') # bar plot comes out wonky in the pngs
        plt.savefig(path+'.svg',format='svg')
