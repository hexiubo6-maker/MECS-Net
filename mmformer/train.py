#coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
import copy
from collections import OrderedDict

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import mmformer
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from predict import AverageMeter, test_softmax
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default='C:\\Users\\xiubo\\Desktop\\brats23', type=str)
parser.add_argument('--dataname', default='BRATS2020', type=str)
parser.add_argument('--savepath', default='C:\\Users\\xiubo\\PycharmProjects\\\hfn\\save', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default='C:\\Users\\xiubo\\PycharmProjects\\hfn\\output\\model_last.pth', type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--disable_contribution', action='store_true', help='Disable the second training stage for contribution to isolate crashes')
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((64,64,64)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks = [[False, False, False, True, True], [False, True, False, False, True], [False, False, True, False, True], [True, False, False, False, True],
         [False, True, False, True, True], [False, True, True, False, True], [True, False, True, False, True], [False, False, True, True, True], [True, False, False, True, True], [True, True, False, False, True],
         [True, True, True, False, True], [True, False, True, True, True], [True, True, False, True, True], [False, True, True, True, True],
         [True, True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())

torch.backends.cudnn.enabled = False

def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print ('dataset is error')
        exit(0)
    model = mmformer.Model(num_cls=num_cls)
    print (model)
    model = torch.nn.DataParallel(model).cuda()

    def optimizer_state_to_cpu(optim):
        # deepcopy to avoid mutating the live optimizer state on GPU
        state = copy.deepcopy(optim.state_dict())
        for v in state.get('state', {}).values():
            for k2, t in list(v.items()):
                if isinstance(t, torch.Tensor):
                    v[k2] = t.detach().to('cpu')
        return state

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True, foreach=False)

    ##########Setting data
    if args.dataname in ['BRATS2020', 'BRATS2015']:
        train_file = 'train.txt'
        test_file = 'test.txt'
    elif args.dataname == 'BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = 'train3.txt'
        test_file = 'test3.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Evaluate
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    # iter_per_epoch = args.iter_per_epoch
    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)
    for epoch in range(args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True
            fuse_pred, sep_preds, prm_preds = model(x, mask)
            torch.cuda.synchronize()

            ### 仅使用前4个类别通道参与损失（与BRATS四类一致）
            fuse_pred4 = fuse_pred[:, 0:4, ...]
            sep_preds4 = [p[:, 0:4, ...] for p in sep_preds]
            prm_preds4 = [p[:, 0:4, ...] for p in prm_preds]

            ###Loss compute
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred4, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred4, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds4:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss

            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds4:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss

            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0+ sep_loss + prm_loss
            else:
                loss = fuse_loss + sep_loss + prm_loss

            optimizer.zero_grad()
            # 形状与数值检查
            try:
                sep_shapes = ', '.join(str(tuple(t.shape)) for t in sep_preds4)
                prm_shapes = ', '.join(str(tuple(t.shape)) for t in prm_preds4)
                logging.info(f"x:{tuple(x.shape)} target:{tuple(target.shape)} fuse:{tuple(fuse_pred4.shape)} sep:[{sep_shapes}] prm:[{prm_shapes}]")
            except Exception:
                pass
            # 非有限loss跳过与梯度裁剪
            if not torch.isfinite(loss.detach()):
                logging.warning(f"Non-finite loss detected at step {step}: {loss.item()} — skipping optimizer step")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.cuda.synchronize()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            logging.info(msg)

            ############ 求贡献度（可选阶段）
            if args.disable_contribution:
                logging.info('[skip contribution stage]')
                continue

            ############ 求贡献度

            Flair_predict = sep_preds[0]
            T1c_predict = sep_preds[1]
            T1_predict = sep_preds[2]
            T2_predict = sep_preds[3]

            Flair_dice = 1
            T1c_dice = 1
            T1_dice = 1
            T2_dice = 1

            Fusion_predict = fuse_pred

            if mask[0][0] == True:
                Flair_dice = criterions.dice_loss(Flair_predict, Fusion_predict, num_cls=num_cls)
            if mask[0][1] == True:
                T1c_dice = criterions.dice_loss(T1c_predict, Fusion_predict, num_cls=num_cls)
            if mask[0][2] == True:
                T1_dice = criterions.dice_loss(T1_predict, Fusion_predict, num_cls=num_cls)
            if mask[0][3] == True:
                T2_dice = criterions.dice_loss(T2_predict, Fusion_predict, num_cls=num_cls)

            min_value = min(Flair_dice, T1c_dice, T1_dice, T2_dice)

            mask_again = [[False, False, False, False, True]]
            if Flair_dice == min_value:
                mask_again[0][0] = True
            if T1c_dice == min_value:
                mask_again[0][1] = True
            if T1_dice == min_value:
                mask_again[0][2] = True
            if T2_dice == min_value:
                mask_again[0][3] = True

            mask_again_cuda = torch.from_numpy(np.array(mask_again)).cuda()

            fuse_pred, sep_preds, prm_preds = model(x, mask_again_cuda)
            torch.cuda.synchronize()

            ###Loss compute
            fuse_pred4 = fuse_pred[:, 0:4, ...]
            sep_preds4 = [p[:, 0:4, ...] for p in sep_preds]
            prm_preds4 = [p[:, 0:4, ...] for p in prm_preds]

            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred4, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred4, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds4:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss

            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds4:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss

            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss + prm_loss
            else:
                loss = fuse_loss + sep_loss + prm_loss

            optimizer.zero_grad()
            # 形状与数值检查（第二阶段）
            try:
                sep_shapes = ', '.join(str(tuple(t.shape)) for t in sep_preds4)
                prm_shapes = ', '.join(str(tuple(t.shape)) for t in prm_preds4)
                logging.info(f"[again] x:{tuple(x.shape)} target:{tuple(target.shape)} fuse:{tuple(fuse_pred4.shape)} sep:[{sep_shapes}] prm:[{prm_shapes}]")
            except Exception:
                pass
            # 非有限loss检查应在反传前
            if not torch.isfinite(loss.detach()):
                logging.warning(f"[again] Non-finite loss detected at step {step}: {loss.item()} — skipping optimizer step")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.cuda.synchronize()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)

            msg = 'Epoch——contribution {}/{}, Iter {}/{}, Loss {:.4f}, moda {}'.format((epoch + 1), args.num_epochs,
                                                                                       (i + 1), iter_per_epoch,
                                                                                       loss.item(), mask_again_cuda)
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            logging.info(msg)

        logging.info('train time per epoch: {}'.format(time.time() - b))

        ##########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        checkpoint = {
            'epoch': int(epoch),
            'state_dict': model.module.state_dict(),
            'optim_dict': optimizer_state_to_cpu(optimizer),
        }
        torch.save(checkpoint, file_name)
        
        if (epoch+1) % 50 == 0 or (epoch>=(args.num_epochs-10)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))

            checkpoint = {
                'epoch': int(epoch),
                'state_dict': model.module.state_dict(),
                'optim_dict': optimizer_state_to_cpu(optimizer),
            }
            torch.save(checkpoint, file_name)

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ##########Evaluate the last epoch model
    test_score = AverageMeter()
    with torch.no_grad():
        logging.info('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            logging.info('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = args.dataname,
                            feature_mask = mask)
            test_score.update(dice_score)
        logging.info('Avg scores: {}'.format(test_score.avg))

if __name__ == '__main__':
    main()
