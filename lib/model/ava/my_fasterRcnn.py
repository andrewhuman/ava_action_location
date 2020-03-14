import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta


class my_faster_rcnn(nn.Module):
    def __init__(self, classes_num, ANCHOR_SCALES,ANCHOR_RATIOS,class_agnostic = True,base_feature_mean = False ,is_add_rpnconv =False,is_pool = True):
        super(my_faster_rcnn,self).__init__()
        self.n_classes = classes_num
        self.ANCHOR_SCALES= ANCHOR_SCALES
        self.ANCHOR_RATIOS= ANCHOR_RATIOS        
        self.class_agnostic = class_agnostic
        self.base_feature_mean = base_feature_mean
        self.is_add_rpnconv = is_add_rpnconv
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_loc = 0
        self.is_pool = is_pool

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model,ANCHOR_SCALES = ANCHOR_SCALES,ANCHOR_RATIOS = ANCHOR_RATIOS,is_add_rpnconv = self.is_add_rpnconv) # rpn之前的特征输出channel

        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()





    def forward(self,  im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        # print('--faster rcnn -- ,im_data = {},im_info = {},gt_boxes = {},num_boxes = {}'.format(im_data.size(), im_info,
        #                                                                                         gt_boxes, num_boxes))

        # feed image data to base model to obtain base feature map
        base_feat_times = self.RCNN_base(im_data) # [1, 832, L/4, 7, 7]
        # print('base_feat_times size = {}'.format(base_feat_times.size()))

        if self.base_feature_mean :
            base_feat_key = torch.mean(base_feat_times,2)
        else:
            base_feat_time_len = base_feat_times.size(2)
            base_feat_key_time = base_feat_time_len // 2
            base_feat_key = base_feat_times[:,:,base_feat_key_time,:,:]
        # base_feat_key_time = 4,base_fear_middle size = torch.Size([1, 832, 7, 7])
        # print('base_feat_times = {} ,base_fear_middle size = {}'.format(base_feat_times.size(),base_feat_key.size()))

        # feed base feature map tp RPN to obtain rois，rois 在proposal layer中已经进行了第一次修正
        # print('im_info = {},gt_boxes = {},num_boxes = {} '.format(im_info,gt_boxes, num_boxes))
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_key, im_info, gt_boxes, num_boxes)
        # rois size = test:([1, 300, 5]) , train:[b, 2000, 5]
        # rois[0,0,:] =[   0.0000,  190.2723,   39.9991,  208.7142,  102.8349] ,[   0.0000,  222.2723,  167.9991,  240.7142,  230.8349]

        #   print('rpn_loss_cls = {}, rpn_loss_cls = {}, rois size = {},rois ={}'.format(rpn_loss_cls,rpn_loss_cls, rois.size(),rois[0,1000,:]))

        # need replicating in time dim for rois

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            #train: rois = ([b, 128, 5]), rois_label = [b*128], rois_target size = torch.Size([b*128, 4])
            #print('---RCNN_proposal_target----,rois = {}, rois_label = {}, rois_target size = {},rois_outside_ws ={}'
            #       .format(rois.size(), rois_label.size(), rois_target.size(), rois_outside_ws.size))

        else:
            
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
        rois = Variable(rois) #[b,max_num,(label,w,h,x,y)],  test: ([b, 300, 5]  train: ([b, 128, 5])

        #recycle roi pooling

        # roi_pooled_fs = []
        # for i in range(base_feat_time_len):
        #
        #     pooled_feat = self.RCNN_roi_pool(base_feat_times[:,:,i,:,:], rois.view(-1, 5))
        #     # print('pooled_feat size = {}'.format(pooled_feat.size()))
        #     torch.c
        #     roi_pooled_fs.append(pooled_feat)
        #
        # print('roi_pooled_fs size = {}'.format(len(roi_pooled_fs)))
        if self.is_pool:
            pooled_feat_0 = self.RCNN_roi_pool(base_feat_times[:, :, 0, :, :], rois.view(-1, 5))  # [b*num,c,w,h] = test:([300, 832, 7, 7]) ,train:  [b*128, 832, 7, 7]
            pooled_feat_0 = torch.unsqueeze(pooled_feat_0,2)
            pooled_feat_1 = torch.unsqueeze(self.RCNN_roi_pool(base_feat_times[:, :, 1, :, :], rois.view(-1, 5)),2)
            pooled_feat_2 = torch.unsqueeze(self.RCNN_roi_pool(base_feat_times[:, :, 2, :, :], rois.view(-1, 5)),2)
            pooled_feat_3 = torch.unsqueeze(self.RCNN_roi_pool(base_feat_times[:, :, 3, :, :], rois.view(-1, 5)),2)
            # pooled_feat_4 = torch.unsqueeze(self.RCNN_roi_pool(base_feat_times[:, :, 4, :, :], rois.view(-1, 5)),2)
            # pooled_feat_5 = torch.unsqueeze(self.RCNN_roi_pool(base_feat_times[:, :, 5, :, :], rois.view(-1, 5)),2)
            # pooled_feat_6 = torch.unsqueeze(self.RCNN_roi_pool(base_feat_times[:, :, 6, :, :], rois.view(-1, 5)),2)
            # pooled_feat_7 = torch.unsqueeze(self.RCNN_roi_pool(base_feat_times[:, :, 7, :, :], rois.view(-1, 5)),2)
            # print('pooled_feat7 size = {},pooled_feat0 size = {}'.format(pooled_feat_7.size(),pooled_feat_0.size()))
        else:
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat_key.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            
            pooled_feat_0  = F.max_pool2d(self.RCNN_roi_crop(base_feat_times[:, :, 0, :, :], Variable(grid_yx).detach()),2,2)
            pooled_feat_0 = torch.unsqueeze(pooled_feat_0,2)
            
            pooled_feat_1  = F.max_pool2d(self.RCNN_roi_crop(base_feat_times[:, :, 1, :, :], Variable(grid_yx).detach()),2,2)
            pooled_feat_1 = torch.unsqueeze(pooled_feat_1,2)
            
            pooled_feat_2  = F.max_pool2d(self.RCNN_roi_crop(base_feat_times[:, :, 2, :, :], Variable(grid_yx).detach()),2,2)
            pooled_feat_2 = torch.unsqueeze(pooled_feat_2,2)
            
            pooled_feat_3  = F.max_pool2d(self.RCNN_roi_crop(base_feat_times[:, :, 3, :, :], Variable(grid_yx).detach()),2,2)
            pooled_feat_3 = torch.unsqueeze(pooled_feat_3,2) 
        
        # test:([b*300, 832,4, 7, 7]) ,train:  [b*128, 832, 4,7, 7]
        pooled_feat_cat =  torch.cat([pooled_feat_0,pooled_feat_1,pooled_feat_2,pooled_feat_3],2) #,pooled_feat_4,pooled_feat_5,pooled_feat_6,pooled_feat_7],2)
        #print('pooled_feat0 size = {} , pooled_feat_cat size = {}'.format(pooled_feat_0.size(),pooled_feat_cat.size()))                     

        #  test: ([b * 300, 1024]),train:[b*128,1024]
        pooled_feat = self._head_to_tail(pooled_feat_cat)
        # print('after top pooled_feat size = {}'.format(pooled_feat.size()))

        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        # print('bbox_pred size = {}'.format(bbox_pred.size()))
        cls_score = self.RCNN_cls_score(pooled_feat)
        # print('cls_score size = {}'.format(cls_score.size()))
        cls_prob = F.softmax(cls_score, 1)
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score,rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        # print('cls_score = {}, bbox_pred = {}'.format(cls_prob,bbox_pred))

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label








    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            m.weight.data.normal_(mean,stddev)
            m.weight.data.zero_()

        # normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)


    def create_architecture(self):
        # print('create_architecture')
        self._init_modules()
        self._init_weights()















