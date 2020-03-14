import torch
from torch import nn
import numpy as np
from myrpn.utils import *
from lib.model.utils.config import cfg

class my_proposal_layer(nn.Module):
    def __init__(self,feat_stride, scales, ratios):
        super(my_proposal_layer,self).__init__()
        self.feat_stride = feat_stride
        self.scales = scales
        self.ratios = ratios
        self.anchor_num = len(self.scales) * len(self.ratios)
        self.anchors = torch.from_numpy(generate_anchors(scales = np.array(scales),ratios = np.array(ratios))).float()


    def forward(self, input):
        """
        :param input:  (rpn_cls_prob.data, rpn_bbox_pred.data,im_info, cfg_key)
        rpn_cls_prob :(b,2*9,w,h)
        rpn_bbox_pred (b,4*9,w,h)

        :return:
        """
        scores = input[0][:,self.anchor_num:,:,:]
        bbox_deltas = input[1]
        im_info = input[2]
        cfg_key = input[3]

        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)





