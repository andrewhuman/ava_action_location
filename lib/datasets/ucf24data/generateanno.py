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
my_anno_file = 'pyannot_{}_correct.pkl'.format(input_frame_num)
PER_IMAGE_MAX_BOX=10

my_result = {}
with open('/home/hyshuai/action_detection/faster-rcnn.pytorch/lib/datasets/ucf24data/annot_origin.pkl' ,'rb') as f:
    database = pickle.load(f)

    for video_name in database.keys():
        label_path_video = os.path.join(label_root,video_name)
        # print('label_path_video : ',label_path_video)
        annotation_one_video = len(database[video_name]['annotations'])

        # 对一个视频里每个tube 进行构建
        video_tube = []
        for ind,anno_tube in enumerate(database[video_name]['annotations']):
            tube_sf = anno_tube['sf']+1 # 1
            tube_ef = anno_tube['ef']-1 # 101 与txt文件的开始，结束对应是正确的
            # print('tube_sf= {},tube_ef = {}  '.format(tube_sf,tube_ef))
            if tube_ef - tube_sf + 1 < input_frame_num :
                # len(anno_tube['boxes'] not correct
                # print('-----empty video_name = {},tube len ={} too small, tube_sf= {},tube_ef = {} '.format(video_name, len(anno_tube['boxes']),tube_sf, tube_ef))
                # print(video_name," this one tube len = ",  len(anno_tube['boxes']),)
                continue
            dict_one_tube = {}
            dict_one_tube['sf'] = tube_sf
            dict_one_tube['ef'] = tube_ef
            boxes_info_tube = []

            # 3 dim list ,compute  each frame
            for frame in range(tube_sf,tube_ef+1,1):
                frame_name = '{:05d}.txt'.format(frame)
                # print('frame: ',frame_name)
                label_txt_real_path = os.path.join(label_path_video, frame_name)
                # print('label_txt_real_path : ', label_txt_real_path)

                # 对每一帧 构建一个2 dim , maybe one or two or three 4, fit to 10
                with open(label_txt_real_path,'rb') as f :
                    # boxes_one_frame_array = np.zeros((PER_IMAGE_MAX_BOX,5),dtype=np.float32)
                    # i = 0
                    boxes_one_frame = []
                    for line in f.readlines():
                        box_info =list(map(np.float32,line.split()))
                        box_xyclass= np.array(box_info[1:]+box_info[:1]) #左上角坐标，右下角左右,,label,

                        # if i < PER_IMAGE_MAX_BOX:
                        #     boxes_one_frame_array[i] = np.array(box_info[1:]+box_info[:1]) #,label,左上角坐标，右下角左右
                        #     i+=1

                        boxes_one_frame.append(box_xyclass)
                    # print('boxes_one_frame_array = ',boxes_one_frame_array,'  shape =  ',boxes_one_frame_array.shape)
                    boxes_info_tube.append(np.asarray(boxes_one_frame))

            if len(boxes_info_tube) < input_frame_num:
                # print('-----------------------------------warning : boxes_info_tube < 16')
                for i in range(input_frame_num - len(boxes_info_tube)):
                    boxes_info_tube.append(np.zeros((1, 5), dtype=np.float32))

            boxes_info_tube_array = np.array(boxes_info_tube)
            # print('video_name = {},tube_sf= {},tube_ef  ={}  shape = {} '.format(video_name,tube_sf, tube_ef,boxes_info_tube_array))
            # print('boxes_info_tube_array = ',boxes_info_tube_array,' shape = ',boxes_info_tube_array.shape)

            dict_one_tube['box_info'] = boxes_info_tube_array

            video_tube.append(dict_one_tube)
        if len(video_tube) != 0:
            # print('video_tube len = {}'.format(len(video_tube)))
            my_result[video_name] = video_tube
        else:
            print('--------error: video is  empty tube in 16 length : ',video_name)



# {'video_name':[{'box_info',[帧数,10,5]},{'box_info',[帧数,10,5]},...],...  }


# {'video_name_0':{'box_info',[帧数,10,5],'sf','ef'   }, }
        # if annotation_one_video > 1 :
        #     print(my_result[video_name]) # first tube,first frame [0]['box_info'][60]
        #     break
print('total len = {}'.format(len(my_result)))
# with open(my_anno_file,'wb') as f:
#
#     pickle.dump(my_result,f,pickle.HIGHEST_PROTOCOL)















