import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import cv2
from ucf24dataset import UCF24Data

# with open('pyannot.pkl' ,'rb') as f:
#     a = pickle.load(f)
#     print(len(a.keys())) # 3194
#
#
#     with open('/data_1/action_detection/ucf24/splitfiles/trainlist01.txt','r') as train_txt:
#         train_list = train_txt.read().splitlines()
#         for train_video_name in train_list:
#             if train_video_name not in a.keys() :
#                 print(' not in anno :',train_video_name)




train_dataset = UCF24Data()
train_data_loader = data.DataLoader(train_dataset,batch_size=1,shuffle=True)

# for data in train_data_loader:
#     rgb_images, box_select = data.
batch_iterator = iter(train_data_loader)

rgb_images, box_select = next(batch_iterator)

print('rgb_images size = ',rgb_images.size(), 'box_select =  ',box_select.size())
