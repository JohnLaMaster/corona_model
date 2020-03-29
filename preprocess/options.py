# Script for running the file
# python preprocess_corona_model.py --path '/home/john/Documents/Research/Covid-19/wetransfer-dfc8e0/case1' --window_center -600 --window_width 1200
# Version 0.0.2

# Pre-processing data for the chinese team's COVID-19 model
import argparse
import os

from pathlib import Path


# Options
class Options():
    def __init__(self):
        super(Options, self).__init__()
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def __check_strings__(self):
        attributes = ['path', 'savedir']
        for attr in attributes:
            if hasattr(self.opt, attr):
                setattr(self.opt, attr, Path(getattr(self.opt, attr))) # Should adjust paths according to operating system
                # check if it is a valid path string, e.g. if it exists
                if os.path.isabs(getattr(self.opt, attr)):
                    return
                elif os.path.isabs(getattr(self.opt, attr)[1:-1]):
                    setattr(self.opt, attr, getattr(self.opt, attr)[1:-1])
                else:
                    print('The given argument is not valid: {}'.format(getattr(self.opt, attr)))
                    return
            else:
                return        

    def initialize(self):
        self.parser.add_argument('--path', type=str, help='path for directory of patient data to be loaded')
        self.parser.add_argument('--savedir', type=str, help='path for exporting output files')
        self.parser.add_argument('--file_ext', type=str, default=['.nii.gz', '_pred.nii.gz'], help='acceptable extensions for files to be loaded')
        self.parser.add_argument('--dir_ext', type=str, default=['202', '203'], help='acceptable directory names for files to be loaded')
        self.parser.add_argument('--add_ext', action='append', dest='file_ext', help='add additional acceptable file extensions')       
        self.parser.add_argument('--threshold', type=float, default=0.30, help='threshold for slice selection')
        self.parser.add_argument('--slice', type=int, default=5, help='select slices every x mm within thresholded range')
        self.parser.add_argument('--no_plotDist', action='store_true', default=False, help='plot and save the segmentation mask distribution and selection per slice')
        self.parser.add_argument('--window_center', type=float, default=300, help='center for windowing Hu image')
        self.parser.add_argument('--window_width', type=float, default=1800, help='width for windowing Hu image')        
        self.parser.add_argument('--segLabel', type=str, default='pred', help='label identifying segmentation files')
        self.parser.add_argument('--d2n', action='store_true', default=False, help='indicates when to process dicom files')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.__check_strings__()

        return self.opt
