from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.ava.my_fasterRcnn import my_faster_rcnn
from model.ava.i3d_net import I3D
from model.ava.pytorch_i3d import InceptionI3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
from collections import OrderedDict

class i3d_rcnn(my_faster_rcnn):
    def __init__(self,classes_num, ANCHOR_SCALES,ANCHOR_RATIOS,class_agnostic=True, pretrained=True,base_feature_mean = False,is_add_rpnconv =False ,is_pool=True):
        self.model_path = '/home/hyshuai/action_detection/faster-rcnn.pytorch/lib/model/ava/rgb_imagenet.pt'
        self.dout_base_model = 832
        self.ANCHOR_SCALES= ANCHOR_SCALES
        self.ANCHOR_RATIOS= ANCHOR_RATIOS
        self.pretrained = pretrained
        self.classes_num = classes_num
        self.base_feature_mean = base_feature_mean
        self.is_add_rpnconv = is_add_rpnconv
        my_faster_rcnn.__init__(self,classes_num=classes_num,ANCHOR_SCALES = ANCHOR_SCALES,ANCHOR_RATIOS = ANCHOR_RATIOS,class_agnostic = class_agnostic,
                                base_feature_mean =  self.base_feature_mean,is_add_rpnconv= self.is_add_rpnconv,is_pool=is_pool)

    def _init_modules(self):
        i3d = I3D()
        if self.pretrained:
            para_old = OrderedDict()
            state_dict = torch.load(self.model_path)

            for name, param in i3d.state_dict().items():
                name_new = name.replace('features.', '').replace('branch0.0', 'b0').replace('branch1.0', 'b1a').replace(
                    'branch1.1', 'b1b') \
                    .replace('branch2.0', 'b2a').replace('branch2.1', 'b2b').replace('branch3.1', 'b3b').replace(
                    'branch2.0', 'b2a').replace('Logits', 'logits.conv3d')
                para_old[name] = state_dict[name_new]

            print("Loading pretrained weights from %s" % (self.model_path))
            i3d.load_state_dict(para_old)

        # Using for rpn
        self.RCNN_base = nn.Sequential(*list(i3d.features._modules.values())[:-6])

        # Using for cls
        self.RCNN_top = nn.Sequential(*list(i3d.features._modules.values())[-5:-3])

        # cls
        self.RCNN_cls_score = nn.Linear(1024, self.classes_num, bias=True)
        self.RCNN_bbox_pred = nn.Linear(1024, 4, bias=True)




    def _head_to_tail(self, after_roi):
        #print('before top after_roi size = {}'.format(after_roi.size())) #(b,1024, 4, 7, 7)
        for_cls = self.RCNN_top(after_roi) 
        # print('after top for_cls size = {}'.format(for_cls.size()))

        for_cls= for_cls.mean(4).mean(3).mean(2) # error have some bug
        return for_cls















