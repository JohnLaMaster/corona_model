from .options import Options
from .processing import PreprocessFiles

# BaseOptions.initialize()
opt = Options().parse()
procedure = PreprocessFiles(opt)

if opt.d2n:
	procedure.convert2nifti()
	procedure.segment()
else:
	procedure.make_dataset()

procedure.ct2jpg()



# consistently check data
# download and preprocess
# Florian, Oliver, Sandeep - preprocessing
