import torch
from mmformer.predict import AverageMeter, test_softmax
from mmformer.data.datasets_nii import Brats_loadall_test_nii
from mmformer.utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
import mmformer

if __name__ == '__main__':
    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
             [True, False, False, False],
             [False, True, False, True], [False, True, True, False], [True, False, True, False],
             [False, False, True, True], [True, False, False, True], [True, True, False, False],
             [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
             [True, True, True, True]]
    mask_name = ['t2', 't1c', 't1', 'flair',
                 't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
                 'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
                 'flairt1cet1t2']

    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    datapath = 'D:/Data_seg/BRATS2020_Training_none_npy'
    test_file = ''
    resume = 'D:/AISegmationCode/mmFormer-main/output/model_last.pth'
    num_cls = 4
    dataname = 'BRATS2020'
