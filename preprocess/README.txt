This file is for preprocessing the data for the Covid-19 project.

File to run the code: preprocess.py
  = Inputs include:
      --path            path for directory of patient data to be loaded
      --savedir         path for exporting output files
      --file_ext        list of acceptable extensions for files to be loaded
      --dir_ext         list of acceptable directory names for files to be loaded
      --add_ext         used to add additional file types to file_ext
      --threshold       threshold for slice selection - should be [0,1]
      --slice           slice is selected eery x mm within the range determined by the threshold
      --no_plotDist     turns off plotting the slice-wise segmentation distribution
      --window_center   center of the Hu window
      --window_width    width of the Hu window
      --segLabel        label identifying segmentation files
      --d2n             indicates when to convert from dicom to nifti
      --jpg             trigger to export data to jpg format

    **Notes:
        - All of these have default values except for the path and savedir
        - savedir is optional. If not specified, the output files will be saved along side the input files
        - threshold could probably still be set higher
        - Currently, a slice is selected every 5mm
        - The Hu window values default to the values for the Chinese team's model
        - jpg defaults to False

If d2n is triggered, that I assume that the data has not been segmented. So this option converts dicom -> nifti and 
segments the data automatically. This can of course be changed by adding a segmentation trigger to the options.

As of Mar 30, if you only want to convert and segment, use --path, --savedir, and --d2n

If the outputs also need to be windowed, resampled, cropped, etc., let me know. It will be easy to add other
functionsalities.
