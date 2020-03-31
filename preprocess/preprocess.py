from options import Options
from processing import PreprocessFiles


opt = Options().parse()
procedure = PreprocessFiles(opt)

if opt.d2n:
	procedure.convert2nifti()
	procedure.segment()

procedure.makeDataset()

if opt.jpg:
	procedure.ct2jpg()
