import torch
from predict import AverageMeter, test_softmax
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
import mmformer
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == '__main__':
    masks = [[False, False, False, True, True], [False, True, False, False, True], [False, False, True, False, True],
             [True, False, False, False, True],
             [False, True, False, True, True], [False, True, True, False, True], [True, False, True, False, True],
             [False, False, True, True, True], [True, False, False, True, True], [True, True, False, False, True],
             [True, True, True, False, True], [True, False, True, True, True], [True, True, False, True, True],
             [False, True, True, True, True],
             [True, True, True, True, True]]

    mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']

    all_masks = [[True, True, True, True, True]]

    all_masks_name = ['flairt1cet1t2']

    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    datapath = 'C:\\Users\\xiubo\\Desktop\\brats23'
    test_file = ''
    resume = 'C:\\Users\\xiubo\\PycharmProjects\\hfn\\save\\model_200.pth'
    num_cls = 4
    dataname = 'BRATS2020'

    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = mmformer.Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])

    test_score = AverageMeter()
    with torch.no_grad():
        print('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            print('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = dataname,
                            feature_mask = mask)
            test_score.update(dice_score)
        print('Avg scores: {}'.format(test_score.avg))