import pickle
import os
import numpy as np
import cv2
#
# img = cv2.imread('e:\\action_detection\\dataset\\ucf24\\rgb-images\\v_RopeClimbing_g23_c02\\00038.jpg')
# cv2.rectangle(img, (44,  50), ( 223, 234),   (0,255,0), 4)
# cv2.imwrite('001_new.jpg', img)

input_frame_num =16
label_root = '/data_1/action_detection/ucf24/labels'
my_anno_file = 'annot_origin.pkl'
testpkl = pickle.loads(open(my_anno_file,'rb').read())
pickle.dump(testpkl,open('annot_origin_py2.pkl','wb'),protocol=2)


# with open('pyannot_16.pkl' ,'rb') as f:
#     anno = pickle.load(f)
#     video_anno = anno['Surfing/v_Surfing_g20_c06']  # [{'s','ef','box ; [[[],]]'}...]
#     print('video_anno len  = ', len(video_anno[1]['box_info']))
#     # tubes_rand_sample = np.random.randint(len(video_anno), size=1)[0]
#
# with open('pyannot.pkl' ,'rb') as f:
#
#     a = pickle.load(f)
#     a_annot = a['SkateBoarding/v_SkateBoarding_g20_c04']
#     print(a_annot)
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