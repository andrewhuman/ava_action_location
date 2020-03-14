# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import torch


from model.ava.i3d_rcnn import i3d_rcnn

from torch.autograd import variable
import torch.utils.data as data
from lib.datasets.ucf24data.ucf24dataset import UCF24Data
from lib.datasets.ucf24data.ucf24dataset_skip import UCF24DataSkip


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--batch_size', dest='batch_size',
                        help='training batch_size',
                        default='1', type=int)
    parser.add_argument('--trained_model', dest='trained_model',
                        help='pretrained_model',
                        default=None, type=str)
    parser.add_argument('--prefix', dest='prefix',
                        help='prefix',
                        default='batch_add_rpn_conv1', type=str)
    parser.add_argument('--lr',dest='lr',default=0.0001,type=float)
    parser.add_argument('--select_box',dest='select_box',help='select_box model',default=1,type=int)
    parser.add_argument('--is_mean',dest='is_mean',help='is to mean to base features, 1 is true,0 is false',
                        default=1,type=int)
    parser.add_argument('--is_pool',dest='is_pool',help='is to use crop or pool',
                        default=1,type=int)    
    parser.add_argument('--anchor_s',dest='anchor_s',
                        help='which anchor to use,2 is scale [2,4,6,8,10,12] and ratios[0.6,1.2,1.8,2.4,3], 3 is [1,2,3,4,5,6,7,8,9,10,11,12,13,14] and [0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.4]',
                        default=2,type=int)    
    parser.add_argument('--max_epoch', dest='max_epoch', help='max_epoch',
                        default=1000, type=int)
    parser.add_argument('--is_add_rpnconv', dest='is_add_rpnconv',
                        help='whether to add rpn conv1, 1 is true,0 is false',
                        default=1, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    batch_size = args.batch_size

    train_dataset = UCF24DataSkip(box_select_mode= args.select_box)
    train_len = len(train_dataset)
    print('train_len = {}, args.prefix = {} '.format(train_len,args.prefix))
    each_epoch = train_len // batch_size
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # a = variable(torch.rand((1,3,16,240,320))).cuda()
    # im_info = variable(torch.from_numpy(np.array([[240,320,16]],dtype=np.float32))).cuda()
    # num_boxes = variable(torch.from_numpy(np.array([[2]],dtype=np.long))).cuda() # (b,k)
    # gt_boxes = variable(torch.from_numpy(np.array([[[10,10,40,40,0],[50,50,150,150,1]]],dtype=np.float32))).cuda()  # (b, K, 5)
    if args.anchor_s == 2:
        ANCHOR_SCALES = [2,4,6,8,10,12]
        ANCHOR_RATIOS = [0.6,1.2,1.8,2.4,3]
    elif args.anchor_s == 3:
        ANCHOR_SCALES = [1,2,3,4,5,6,7,8,9,10,11,12,13,14] 
        ANCHOR_RATIOS = [0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.4]
    elif args.anchor_s == 4:
        ANCHOR_SCALES = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18] 
        ANCHOR_RATIOS = [0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.4,3.7]    

    net = i3d_rcnn(classes_num=25, ANCHOR_SCALES = ANCHOR_SCALES,ANCHOR_RATIOS = ANCHOR_RATIOS,base_feature_mean=(args.is_mean == 1), 
                   is_add_rpnconv=(args.is_add_rpnconv == 1), is_pool=(args.is_pool == 1))
    net.create_architecture()
    if args.trained_model is not None:
        print('loading pretrained model {}'.format(args.trained_model))
        load_model = torch.load(args.trained_model)['model']
        net.load_state_dict({k.replace('module.', ''): v for k, v in load_model.items()})
    # print(net)
    net.cuda()
    # net = torch.nn.DataParallel(net)

    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=5e-4)

    for step in range(args.max_epoch):
        batch_iterator = iter(train_data_loader)
        for epoch in range(each_epoch):
            start = time.time()

            # rgb_images = [batch, 3, 16, 240, 320],boxes
            rgb_images, boxes, num_box = next(batch_iterator)

            input_img = variable(rgb_images).cuda()
            im_info = variable(torch.from_numpy(np.array([[240, 320, 16]] * batch_size, dtype=np.float32))).cuda()
            num_boxes = variable(num_box).cuda()
            gt_boxes = variable(boxes).cuda()

            net.zero_grad()
            rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = net(
                input_img, im_info, gt_boxes, num_boxes)
            fg_cnt = torch.sum(rois_label.data.ne(0))

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('total step = {}, epoch = {}, time = {}  ---------------------------------------------------------------------fg ={}'.format(step,epoch, (str((time.time() - start) * 1000)), fg_cnt))
            print('rpn_loss_cls = {} rpn_loss_box = {}, RCNN_loss_cls = {} RCNN_loss_bbox = {} total loss ={}'.format(
                rpn_loss_cls.item(),
                rpn_loss_box.item(),
                RCNN_loss_cls.item(),
                RCNN_loss_bbox.item(),
                loss.item()))
            print('--------------------------------- --------------------------------------------------------------')
            # print('epoch time = {} loss --------------------------------------------------------------= {} '.format((str((time.time() - start) * 1000)), loss.item()))

        if step % 2 == 0 or step == (args.max_epoch -1):
            save_name = 'faster_rcnn_mine_{}_{}.pth'.format(args.prefix,step)

            torch.save({
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': step
            }, save_name)

        # print('----------------------------------------------')
        # print('----------------------------------------------')

