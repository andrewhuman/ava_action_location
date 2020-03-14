from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        rpn_cls_score = input[0]
        # print('---_AnchorTargetLayer---  rpn_cls_score size  anchor w  h = {}'.format(rpn_cls_score.size()))
        gt_boxes = input[1]
        im_info = input[2]
        num_boxes = input[3]

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        batch_size = gt_boxes.size(0)

        # 1. Generate proposals from bbox deltas and shifted anchors
        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3) # 15,20
        shift_x = np.arange(0, feat_width) * self._feat_stride # # feat_stride = 16,[0,16,32,...,320]
        shift_y = np.arange(0, feat_height) * self._feat_stride  # feat_stride = 16,[0,16,32,...,240]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())

        # [7 * 7 = 49,4]
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        # print('---_AnchorTargetLayer---  shifts = {}'.format(shifts))

        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.
        # [[-15.,  -4.,  30.,  19.],
        # [-38., -16.,  53.,  31.],
        # [-84., -40.,  99.,  55.],
        # [ -8.,  -8.,  23.,  23.],
        # [-24., -24.,  39.,  39.],
        # [-56., -56.,  71.,  71.],
        # [ -3., -14.,  18.,  29.],
        # [-14., -36.,  29.,  51.],
        # [-36., -80.,  51.,  95.]]

        # [[-84., -40., 99., 55.],
        #  [-176., -88., 191., 103.],
        #  [-360., -184., 375., 199.],
        #  [-56., -56., 71., 71.],
        #  [-120., -120., 135., 135.],
        #  [-248., -248., 263., 263.],
        #  [-36., -80., 51., 95.],
        #  [-80., -168., 95., 183.],
        #  [-168., -344., 183., 359.]]
        # print('---_AnchorTargetLayer---  _anchors = {}'.format(self._anchors))
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)
        # all_anchors = [441, 4] = [15 * 20 * 25 ,4 ]

        total_anchors = int(K * A)
        # print('---_AnchorTargetLayer---  all_anchors num  = {}'.format(all_anchors.size()))
        # print('---_AnchorTargetLayer---  all_anchors = {} '.format(all_anchors[-100:]))


        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))
        # inds_inside = 210
        inds_inside = torch.nonzero(keep).view(-1)
        # print('---_AnchorTargetLayer---  inds_inside = {}'.format(inds_inside.size()))

        # keep only inside anchors 202...
        anchors = all_anchors[inds_inside, :]
        # print('---_AnchorTargetLayer---  keep_anchors = {} '.format(anchors[2000:2100]))

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

        # [overlaps = b ,N(anchor) ,K(box)   ]
        # if gt_boxes 为[0,0,0,0,0]  overlaps会置位0，anchor若为0，overlap置位-1
        # print('---_AnchorTargetLayer---  gt_boxes = {}'.format(gt_boxes))
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)  # 计算所有anchors和gt boxes的覆盖率 [b,N = anchor num  ,k = gt_box num]
        # print('---_AnchorTargetLayer---  overlaps = {}'.format(overlaps))

        # [max_overlaps = b ,N(anchor)    ]
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)  # 每个anchors最大overlap对应gt boxes的index的overlap值,#每个anchors最大的overlap对应gt boxes的index
        # [gt_max_overlaps = b ,K(box)    ]
        gt_max_overlaps, _ = torch.max(overlaps, 1)  # 每个gt box最大的overlap对应的anchor的overlap值 # [b, k = gt_box num  ]
        # print('---_AnchorTargetLayer---  max_overlaps shape = {}'.format(max_overlaps.size())) # [b,n=anchor num,]
        # print('---_AnchorTargetLayer---  argmax_overlaps = {}'.format(argmax_overlaps))
        # print('---_AnchorTargetLayer---  gt_max_overlaps.size()= {}'.format(gt_max_overlaps.size()))
        # print('---_AnchorTargetLayer---  gt_max_overlaps= {}'.format(gt_max_overlaps))

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0  # 每个anchors和真值覆盖率小于cfg.TRAIN.RPN_NEGATIVE_OVERLAP的为负样本

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)
        # print('---_AnchorTargetLayer---  keep= {}'.format(keep))

        if torch.sum(keep) > 0:
            labels[keep>0] = 1  # 最大覆盖率的anchors为正样本

        # fg label: above threshold IOU
        # print('---_AnchorTargetLayer---  max_overlaps > {} ========================== {}'.format(cfg.TRAIN.RPN_POSITIVE_OVERLAP,torch.sum((max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP).int(),1).data))
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1  # 最大覆盖率的anchors为正样本
        #print('cfg.TRAIN.RPN_POSITIVE_OVERLAP = {}'.format(cfg.TRAIN.RPN_POSITIVE_OVERLAP))

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)  # 选择正负样本，总共256个，默认正负样本各一半，正样本不足时候负样本pad

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive labels if we have too many 去掉多余的
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

#           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many 去掉多余的
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        # 计算box regression的偏移量 返回差值，dx，dy，dw，dh，即每个anchor与它overlap最大的gt_box之间的偏移量
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # use a single value instead of 4 values for easy index.
        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]  # 只有正样本有inside_weights, = 1.0

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()  # 1 / 256
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        # print('positive_weights={}, negative_weights = {} '.format(positive_weights,negative_weights))
        bbox_outside_weights[labels == 1] = positive_weights  # 正负样本都有outside_weights = 1 / 256
        bbox_outside_weights[labels == 0] = negative_weights  # 正负样本都有outside_weights

        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        print( '---_AnchorTargetLayer---  ---------------------------------labels[0] == 1 sum = {}, sum_fg = {}'.format((torch.sum(labels[0] == 1)),sum_fg ))
        # print('labels[0] == 0 sum = {},sum_bg={}'.format((torch.sum(labels[0] == 0)),sum_bg))
        outputs.append(labels) # 输出label

        bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        outputs.append(bbox_targets)  # 输出差

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()

        outputs.append(bbox_inside_weights)  # 输出inside_weights

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)  # 输出 utside_weights

        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
