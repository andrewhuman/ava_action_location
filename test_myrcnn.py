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
import cv2
import torch
from lib.model.utils.config import cfg
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.nms.nms_wrapper import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.utils.net_utils import  vis_detections

from lib.model.ava.i3d_rcnn import i3d_rcnn

from torch.autograd import variable
import torch.utils.data as data
from lib.datasets.ucf24data.ucf24dataset import UCF24Data
from lib.datasets.ucf24data.ucf24dataset_skip import UCF24DataSkip

class_name = ['__background__','Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling','Diving','Fencing', #8
              'FloorGymnastics','GolfSwing','HorseRiding','IceDancing','LongJump','PoleVault','RopeClimbing','SalsaSpin',#16
              'SkateBoarding','Skiing','Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping','VolleyballSpiking',
              'WalkingWithDog']

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--batch_size', dest='batch_size',
                        help='training batch_size',
                        default='1', type=int)
    parser.add_argument('--thresh_class', dest='thresh_class',
                        help='training batch_size',
                        default='0.8', type=float)    
    parser.add_argument('--trained_model', dest='trained_model',
                        help='pretrained_model',
                        default=None, type=str)
    parser.add_argument('--select_box',dest='select_box',help='select_box model',default=1,type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--is_mean',dest='is_mean',help='is to mean to base features, 1 is true,0 is false',
                        default=1,type=int)
    parser.add_argument('--is_add_rpnconv', dest='is_add_rpnconv', help='whether to add rpn conv1, 1 is true,0 is false',
                        default=1, type=int)
    parser.add_argument('--is_pool',dest='is_pool',help='is to use crop or pool',
                        default=1,type=int)
    parser.add_argument('--BBOX_REG',dest='BBOX_REG',help='BBOX_REG',
                        default=1,type=int)        
    parser.add_argument('--anchor_s',dest='anchor_s',
                        help='which anchor to use,2 is scale [2,4,6,8,10,12] and ratios[0.6,1.2,1.8,2.4,3], 3 is [1,2,3,4,5,6,7,8,9,10,11,12,13,14] and [0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.4]',
                        default=2,type=int)      
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    batch_size = args.batch_size
    class_num = 25
    thresh = 0.05
    vis = True

    #train_dataset = UCF24Data(is_train=False,video_list_txt = '/data_1/action_detection/ucf24/splitfiles/testlist01.txt') #
    train_dataset = UCF24DataSkip(is_train = False,box_select_mode= args.select_box,video_list_txt = '/home/hyshuai/action_detection/faster-rcnn.pytorch/lib/datasets/ucf24data/test_list_mine.txt',)
    train_len = len(train_dataset)
    print('train_len = ', train_len)
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
    

    #net = i3d_rcnn(classes_num=25,base_feature_mean=(args.is_mean==1),is_add_rpnconv = (args.is_add_rpnconv==1))
    net.create_architecture()
    if args.trained_model is not None:
        print('loading pretrained model {}'.format(args.trained_model))
        load_model = torch.load(args.trained_model)['model']
        net.load_state_dict({k.replace('module.', ''): v for k, v in load_model.items()})



    #net.load_state_dict({k.replace('module.',''):v for k,v in load_model.items()})
    # print(net)
    net.cuda()
    net.eval()

    batch_iterator = iter(train_data_loader)
    for epoch in range(100):
        det_tic = time.time()

        # rgb_images = [batch, 3, 16, 240, 320],boxes
        rgb_images, boxes, num_box ,img_filelist = next(batch_iterator)

        input_img = variable(rgb_images).cuda()
        im_info = variable(torch.from_numpy(np.array([[240, 320, 16]] * batch_size, dtype=np.float32))).cuda()
        num_boxes = variable(num_box).cuda()
        gt_boxes = variable(boxes).cuda()

        # rois = torch.Size([1, 300, 5]), cls_prob = torch.Size([1, 300, 25]), bbox_pred = torch.Size([1, 300, 4])
        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = net(
            input_img, im_info, gt_boxes, num_boxes)

        #print('rois = {}, cls_prob = {}, bbox_pred = {},'.format(rois.size(), cls_prob.size(),bbox_pred.size()))
        #print('score = {} '.format(cls_prob.data[0,80:100,:]))
        #print('max score = {} '.format(torch.max(cls_prob[0],1)))
        # print('arg max score = {} '.format(torch.argmax(cls_prob[0], 1)))
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if args.BBOX_REG == 1:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            p = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
            pred_boxes = boxes.cuda()
        #print('pred_boxes = {} '.format(pred_boxes.size()))




        # pred_boxes /= data[1][0][2].item() divide input and origin image scale
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        print(img_filelist)
        im = cv2.imread(img_filelist[8][0])
        im2show = np.copy(im)

        # for j in range(1, class_num):
        #     inds = torch.nonzero(scores[:, j] > thresh).view(-1)
        #     # if there is det
        #     if inds.numel() > 0:
        #         cls_scores = scores[:, j][inds]
        #         _, order = torch.sort(cls_scores, 0, True)
        #         if args.class_agnostic:
        #             cls_boxes = pred_boxes[inds, :]
        #         else:
        #             cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
        #
        #         cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        #         # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
        #         cls_dets = cls_dets[order]
        #         keep = nms(cls_dets, cfg.TEST.NMS)
        #         cls_dets = cls_dets[keep.view(-1).long()]
        #         if vis:
        #             im2show = vis_detections(im2show, class_name[j], cls_dets.cpu().numpy(), 0.3)
        #         all_boxes[j][i] = cls_dets.cpu().numpy()
        #     else:
        #         all_boxes[j][i] = empty_array

        for j in range(1, class_num):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            inds_threshold = torch.nonzero(scores[:,j]>0.5).view(-1)
            print('inds > 0.05 num = {},inds_threshold > 0.5 ={}      {}'.format(inds.numel(),inds_threshold.numel(),class_name[j]))
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[inds, :]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, 0.2, force_cpu=not cfg.USE_GPU_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, class_name[j], cls_dets.cpu().numpy(), args.thresh_class)
        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
        
        prefix_img= img_filelist[8][0].split("/")[5]
        result_path = 'result/{}_{}.jpg'.format(prefix_img,str(epoch))
        cv2.imwrite(result_path, im2show)




