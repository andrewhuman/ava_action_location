import pickle
import os
import numpy as np



#
input_frame_num =16
tube_skip_length =32
label_root = '/data_1/action_detection/ucf24/labels'
my_anno_file = 'anno_tube_{}_2.pkl'.format(input_frame_num)
PER_IMAGE_MAX_BOX=10

train_origin_list =[]
train_txt = '/data_1/action_detection/ucf24/splitfiles/trainlist01.txt'
with open(train_txt,'r') as f:
    for line in f.readlines():
        name = str(line.split()[0])
        print(name)
        train_origin_list.append(name)
        

test_origin_list =[]
test_txt = '/data_1/action_detection/ucf24/splitfiles/testlist_backup.txt'
with open(test_txt,'r') as f:
    for line in f.readlines():
        name = str(line.split()[0])
        print(name)
        test_origin_list.append(name)
        

train_new_list =[]
test_new_list =[]
#
my_result = {}

with open('/home/hyshuai/action_detection/faster-rcnn.pytorch/lib/datasets/ucf24data/annot_origin.pkl' ,'rb') as f:
    database = pickle.load(f)

    for video_name in database.keys():
        label_path_video = os.path.join(label_root,video_name)
        # print('label_path_video : ',label_path_video)
        annotation_one_video = len(database[video_name]['annotations'])

        # 对一个视频里每个tube 进行构建
        video_tube = []
        for index_tube,anno_tube in enumerate(database[video_name]['annotations']):
            tube_sf = anno_tube['sf']+1 # 1
            tube_ef = anno_tube['ef']-1 # 101 与txt文件的开始，结束对应是正确的
            tube_length = tube_ef - tube_sf + 1
            # print('tube_sf= {},tube_ef = {}  '.format(tube_sf,tube_ef))
            if tube_length < input_frame_num :
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
                print('-----------------------------------warning : boxes_info_tube < 16')
                for i in range(input_frame_num - len(boxes_info_tube)):
                    boxes_info_tube.append(np.zeros((1, 5), dtype=np.float32))
            assert len(boxes_info_tube) == tube_length

            # ignore if two tube is same
            dont_generate_tube =False
            for generated_tube in video_tube:
                if abs(generated_tube['sf'] - dict_one_tube['sf'])+abs(generated_tube['ef'] - dict_one_tube['ef']) < 9:
                    print('{}: generated_tube sf = {}, dict_one_tube sf= {},generated_tube ef= {} ,dict_one_tube ef= {} '
                          .format(video_name,generated_tube['sf'], dict_one_tube['sf'], generated_tube['ef'],dict_one_tube['ef']))
                    dont_generate_tube =True
                    break
            if dont_generate_tube:continue

            # if not ignore ,generate tube
            tube_name = str(video_name+'_'+str(index_tube))
            for start in range(0,tube_length, tube_skip_length):
                if tube_length-start > input_frame_num:
                    print('tube_length = {}, start index = {},end index = {}'.format(tube_length,start,(start+tube_skip_length)))
                    
                    if  video_name in train_origin_list:
                        train_new_list.append([tube_name,start,tube_length])
                    else:
                        test_new_list.append([tube_name,start,tube_length])

            boxes_info_tube_array = np.array(boxes_info_tube)
            # print('video_name = {},tube_sf= {},tube_ef  ={}  shape = {} '.format(video_name,tube_sf, tube_ef,boxes_info_tube_array))
            # print('boxes_info_tube_array = ',boxes_info_tube_array,' shape = ',boxes_info_tube_array.shape)

            dict_one_tube['box_info'] = boxes_info_tube_array
            my_result[tube_name] = dict_one_tube
            video_tube.append(dict_one_tube)


        print('{} right: video tube num = {} '.format(video_name,len(video_tube)))
        if len(video_tube) == 0:
            print('---------------error: video all tube is small then  16 length : ', video_name)
            # print('video_tube len = {}'.format(len(video_tube)))
            # my_result[video_name] = video_tube
        # else:
        #     print('--------error: video is  empty tube in 16 length : ',video_name)
print('train_new_list train_new_list = {}'.format(len(train_new_list)))
print('test_new_list test_new_list = {}'.format(len(test_new_list)))
print('total len = {}'.format(len(my_result)))

# {'video_name':[{'box_info',[帧数,10,5]},{'box_info',[帧数,10,5]},...],...  }


# {'video_name_0':{'box_info',[帧数,10,5],'sf','ef'   }, }
        # if annotation_one_video > 1 :
        #     print(my_result[video_name]) # first tube,first frame [0]['box_info'][60]
        #     break
with open('train_list_mine.txt','w') as f:
            for tube_skip in train_new_list:
                f.write(str(tube_skip[0])+"\t"+str(tube_skip[1])+"\t"+str(tube_skip[2])+"\n")
                
with open('test_list_mine.txt','w') as f:
    for tube_skip in test_new_list:
        f.write(str(tube_skip[0])+"\t"+str(tube_skip[1])+"\t"+str(tube_skip[2])+"\n")
        

with open(my_anno_file,'wb') as f:

    pickle.dump(my_result,f,pickle.HIGHEST_PROTOCOL)















