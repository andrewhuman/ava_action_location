# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

import pickle
import os
import numpy as np
# with open('pyannot.pkl' ,'rb') as f:
#
#     a = pickle.load(f)
#     print(len(a.keys())) # # 3194
#     # for ke in a.keys():
#     #     l = len(a[ke]['annotations'])
#     #     print(ke)
#     #
#     #     for i in range(l):
#     #         en_st = a[ke]['annotations'][i]['ef'] - a[ke]['annotations'][i]['sf']
#     #         print('st = {}, end = {} ,leng = {} '.format(a[ke]['annotations'][i]['sf'], a[ke]['annotations'][i]['ef'],
#     #                                                      en_st))
#
#             # if l > 1 and a[ke]['annotations'][1]['sf'] <= a[ke]['annotations'][0]['ef']:
#         #     print(a[ke]['annotations'])
#         #     print(ke)
#
#     # print(len(a.keys()))
#     boxs = a['PoleVault/v_PoleVault_g25_c02']['annotations'][0]
#     print( boxs)
    # print(a['SkateBoarding/v_SkateBoarding_g09_c01']['annotations'][0]['label'])
    # print(a['SkateBoarding/v_SkateBoarding_g09_c01']['annotations'][0]['boxes'])
    # print(a['SkateBoarding/v_SkateBoarding_g09_c01']['annotations'][0]['ef'])


#
input_frame_num =16
label_root = '/data_1/action_detection/ucf24/labels'
my_anno_file = 'pyannot_{}_clear.pkl'.format(input_frame_num)

my_result = {}
total_scale = []
with open('/home/hyshuai/action_detection/faster-rcnn.pytorch/lib/datasets/ucf24data/annot_origin_py2.pkl' ,'rb') as f:
    database = pickle.load(f)

    for video_name in database.keys():
        label_path_video = os.path.join(label_root,video_name)
        # print('label_path_video : ',label_path_video)
        annotation_one_video = len(database[video_name]['annotations'])


        # 对一个视频里每个tube 进行构建
        video_tube = []
        mean_video = []
        for ind,anno_tube in enumerate(database[video_name]['annotations']):
            tube_sf = anno_tube['sf']+1 # 1
            tube_ef = anno_tube['ef']-1 # 101
            # print('tube_sf= {},tube_ef = {}  '.format(tube_sf,tube_ef))
            if tube_ef - tube_sf  < input_frame_num :
                print(video_name," one tube len = ",  len(anno_tube['boxes']))
                continue
            dict_one_tube = {}
            dict_one_tube['sf'] = tube_sf
            dict_one_tube['ef'] = tube_ef
            boxes_info_tube = []

            # 3 dim list ,compute  each frame
            for frame in range(tube_sf,tube_ef,1):
                frame_name = '{:05d}.txt'.format(frame)
                # print('frame: ',frame_name)
                label_txt_real_path = os.path.join(label_path_video, frame_name)
                # print('label_txt_real_path : ', label_txt_real_path)

                # 对每一帧 构建一个2 dim , maybe one or two or three
                with open(label_txt_real_path,'rb') as f :
                    boxes_info_frame = []
                    for line in f.readlines():
                        box_info =list(map(float,line.split()))
                        box_info_whxyl = np.array(box_info[1:]+box_info[:1]) #左上角坐标，右下角左右,label

                        boxes_info_frame.append(box_info_whxyl)
                        scale =(box_info[4] -box_info[2]) * 1.0   /  (box_info[3] -box_info[1])
                        # scale =np.sqrt((box_info[4] -box_info[2]) * 1.0  *  (box_info[3] -box_info[1]) / 224)
                        mean_video.append(scale)

                    boxes_info_frame_array = np.array(boxes_info_frame)
                    # print('boxes_info_frame_array = ',boxes_info_frame_array,'  shape =  ',boxes_info_frame_array.shape)
                    boxes_info_tube.append(boxes_info_frame_array)

            boxes_info_tube_array = np.array(boxes_info_tube)
            # print('boxes_info_tube_array = ',boxes_info_tube_array,' shape = ',boxes_info_tube_array.shape)

            dict_one_tube['box_info'] = boxes_info_tube_array

            video_tube.append(dict_one_tube)
        if len(video_tube) != 0:
            my_result[video_name] = video_tube
        else:
            print('error: 16 length empty tube : ',video_name)

        # 对一个视频求平均
        mean = np.mean(np.array(mean_video))
        total_scale.append(mean)
#         print('{} : {}'.format(video_name,mean))
        print(mean)



        # if annotation_one_video > 1 :
        #     print(my_result[video_name]) # first tube,first frame [0]['box_info'][60]
        #     break

# with open(my_anno_file,'wb') as f:
#
#     pickle.dump(my_result,f,pickle.HIGHEST_PROTOCOL)



plt.figure(figsize=(1000,2))

# my_y_ticks = np.arange(-0.1, 0.1, 0.01)
# plt.yticks(my_y_ticks)


plt.scatter(x =total_scale,y = np.ones_like(total_scale),s=75,c =100,  alpha=.5)
plt.show()

plt.hist(total_scale, bins=50, color='steelblue', normed=True )












