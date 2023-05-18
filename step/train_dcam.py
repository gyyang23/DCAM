from core.cls_dcam_trainer import ClsDcamTrainer
from misc import torchutils
from misc.focalloss import FocalLoss

import os
import torch
from torch import nn
from torch.utils.data import DataLoader

import voc12.dataloader
import importlib


def run(args):
    # ----------------------------------------
    # initial
    # ----------------------------------------
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('TRAIN ON DEVICE:', device)

    # data
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    data_loaders = {'train': train_data_loader, 'val': val_data_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    max_step = (len(train_dataset) // args.cam_batch_size) * args.dcam_num_epoches

    # model
    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    print('module:%s' % (model.backbone[3][0][2].conv3))
    modules = [model.backbone[3][0][2].conv3]

    # train
    criterion = nn.MultiLabelSoftMarginLoss()

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': 0.1 * args.dcam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 0.1 * args.dcam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.dcam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    # ----------------------------------------
    # train
    # ----------------------------------------
    model_path = os.path.join(args.cam_weight_dir, 'checkpoint.pth')
    result_path = os.path.join(args.dcam_weight_dir)

    print('-' * 40)
    print('CHECK RESULT PATH:', result_path)
    print('-' * 40)
    trainer = ClsDcamTrainer(model=model,
                             modules=modules,
                             criterion=criterion,
                             device=device,
                             optimizer=optimizer,
                             data_loaders=data_loaders,
                             dataset_sizes=dataset_sizes,
                             num_epochs=args.dcam_num_epoches,
                             result_path=result_path,
                             model_path=model_path
                             )

    trainer.train()
    trainer.check()


if __name__ == '__main__':
    run()
