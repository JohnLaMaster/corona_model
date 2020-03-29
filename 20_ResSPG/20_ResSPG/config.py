import warnings
from utils.anchor import get_anchor


class DefaultConfig(object):
    root = '/data2/users/hzw/NTS'
    batch_size = 8
    num_classes = 2
    threshold = 0.6
    lr = 0.002
    weight_decay = 0.9
    max_epoch = 60
    num_workers = 32
    input_size = (448, 448)#(356, 356)
    crop_size = (448, 448)
    model_path = ''#'./checkpoints/fourth002.pkl'
    scales = [64, 128, 160, 192, 256]
    stride = 48
    ratio = [0.5, 1, 1.5, 2, 3]
    save_dir = './save_bins'
    anchor = get_anchor(ratio, scales)

    save_atten_epoch = 20

    gpu = '0,1'

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()
