# Version 0.0.2

# Pre-processing data for the chinese team's COVID-19 model
import os
import shlex

import numpy as np
from scipy import ndimage
from skimage.transform import rescale


# Auxiliary functions
def make_dataset(dir, opt):
    dataset, segmentation = [], []
    file_ext = tuple(opt.file_ext)
    if os.path.exists(dir):
        assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    # print(sorted(os.walk(dir)))
    for root, dirs, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if not fname.startswith('._'):
                if fname.endswith(file_ext):
                    path = os.path.join(root, fname)
                    if opt.segLabel in fname:
                        segmentation.append(path)
                    else:
                        dataset.append(path)

    return zip(sorted(dataset), sorted(segmentation))

def convertdcm(dir, opt):
    dataset = []
    dirs_ext = tuple(opt.dirs_ext)   
    if os.path.exists(dir):
        assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    if opt.d2n:
        for root, dirs, fnames in sorted(os.walk(dir)):
            if root.endswith(dir_ext):
                base, part2 = os.path.split(root)
                base, part1 = os.path.split(base)
                _, part0 = os.path.split(base)
                outname = part0 + '_' + part1 + '_' + part2
                readableCmd = 'dcm2niix -f {} -i {} -s n -m y -z y -o {}'.format(outname,root,opt.savedir)
                command = shlex.split(readableCmd)

                try:
                    logfilepath = os.path.join(opt.savedir, outname + "_conversion.log")
                    with open(logfilepath, "w") as outfile:
                        subprocess.run(
                            command, stdout=outfile, stderr=outfile)
                    dataset.append(os.path.join(opt.savedir,outname))
                except Exception as e:
                    print('error: ' + str(e))
                    print("conversion error for:", root)

    return dataset

def transform_to_hu(medical_image, image):
    intercept = medical_image['scl_inter'] if not isnan(medical_image['scl_inter']) else 0
    slope = medical_image['scl_slope'] if not isnan(medical_image['scl_slope']) else 1
    hu_image = image * slope + intercept

    return hu_image

def window_image(image, window_center=300, window_width=1800):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image = np.clip(window_image, img_min, img_max)
    
    return window_image, img_min, img_max

def isnan(input):
    return True if (input!=input) else False

def resample(image, image_thickness, pixel_spacing):
    new_size = np.array([1, 1, 1])
    
    x_pixel = float(pixel_spacing[0])
    y_pixel = float(pixel_spacing[1])
    
    size = np.array([x_pixel, y_pixel, float(image_thickness)])
    
    image_shape = np.array([image.shape[0], image.shape[1], image.shape[2]])
    
    new_shape = image_shape * size * new_size
    new_shape = np.round(new_shape)
    resize_factor = new_shape / image_shape
    
    # resampled_image = ndimage.interpolation.zoom(image, resize_factor)    # I read as of 2017, this had a lot of bugs. If it's been fixed, then it's another option
    resampled_image = rescale(image, resize_factor, anti_aliasing=True)
    
    return resampled_image

def normalize(input, min, max, low, high):
    mult = high - low
    out = ((input - min) / (max - min)) * mult + low

    return out
