from .options import Options
from .processing import PreprocessFiles

# BaseOptions.initialize()
opt = Options().parse()
procedure = PreprocessFiles(opt)

procedure.process()
