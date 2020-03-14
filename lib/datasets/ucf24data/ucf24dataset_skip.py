import torch
from torchvision import transforms as tvtsf
import torch.utils.data as data
import os
import pickle
import numpy as np
import cv2
from lib.datasets.ucf24data import util
from model.utils.config import cfg

def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


class Transform(object):
    def __init__(self, min_size=320, max_size=400):
        super(Transform,self).__init__()
        # self.min_size = min_size
        # self.max_size = max_size

    def __call__(self,in_data):
        img, bbox,flip,crop ,is_train= in_data # bbox = [num,5]
        C, H,W= img.shape
        # print( C, H,W) # 3 240 320
        img = (img / 255.) * 2 - 1
        MAX_BOXES_NUM = 10
        box_final = np.zeros((MAX_BOXES_NUM, 5), dtype=np.float32)  # 10,5
        box_se_len = min(MAX_BOXES_NUM, len(bbox)) # box bum


        # horizontally flip
        if flip and not is_train:
            # print('before flip img[0,120,10] = {},bbox = {}'.format(img[0,120,10],bbox) )
            img = util.flip_img(img)
            bbox= util.flip_box(bbox,(H,W))
            # print('after flip img[0,120,10] = {},bbox = {}'.format(img[0, 120, 10], bbox))
            # img, params = util.random_flip(
            #     img, x_random=True, return_param=True)
            # bbox = util.flip_bbox(
            #     bbox, (o_H, o_W), x_flip=params['x_flip'])
        

        for i in range(len(bbox)):
            if bbox[i][0] < 0 or bbox[i][1] < 0 or bbox[i][2] >=W or bbox[i][3] >= H or bbox[i][0] > bbox[i][2] \
                    or bbox[i][1] > bbox[i][3]:
                print('box error : ',bbox[i])

        box_final[:box_se_len] = bbox[:box_se_len]
        # print('box_se_len ={} ,box_final = {}'.format(box_se_len, box_final))


        return img, box_final


class UCF24DataSkip(data.Dataset):

    def __init__(self,

                 video_list_txt = '/home/hyshuai/action_detection/faster-rcnn.pytorch/lib/datasets/ucf24data/train_list_mine.txt',
                 anno_file = '/home/hyshuai/action_detection/faster-rcnn.pytorch/lib/datasets/ucf24data/anno_tube_16_2.pkl',
                 img_path = '/data_1/action_detection/ucf24/rgb-images',
                 is_train = True,
                 input_frame_num=16,
                 train_list_skip =32,
                 box_select_mode=1
                 ):
        """
        :param args:  args params ,root_path dataset = 'ucf24',

        """
        self.img_path = img_path
        self.input_frame_num = input_frame_num
        self.train_list_skip = train_list_skip
        self.box_select_mode = box_select_mode
        with open(video_list_txt,'r') as train_list:
            self.video_list = train_list.read().splitlines()
            self.video_len = len(self.video_list)


        with open(anno_file,'rb') as my_anno:
            self.anno = pickle.load(my_anno)  # # {'video_name_0':{'box_info',[帧数,10,5],'sf','ef'   }, }
            # {'video_name':[{'box_info',[帧数,10,5],'sf',ef'},{'box_info',[帧数,10,5]},...],...  }
        self.tsf = Transform()
        self.is_train = is_train


    def __getitem__(self, item):
        tube_info = self.video_list[item].split()
        video_name = str(tube_info[0])
        start_index = int(tube_info[1])
        tube_length = int(tube_info[2])
        # video_name = self.video_list[item]
        print('video_name =', video_name)

        real_anno = self.anno[video_name] # {'box_info',[帧数,10,5],'sf','ef'   }
        # print('video_anno = ', video_anno)
        # tubes_rand_sample = np.random.randint(len(video_anno),size=1)[0]
        # print( 'len(video_anno) =  ',len(video_anno), 'tubes_rand_sample = ', tubes_rand_sample)

        # real_anno  = video_anno[tubes_rand_sample]

        sf = real_anno['sf']
        ef = real_anno['ef'] # and start_index ,total_length
        box_infos = real_anno['box_info']  # [帧数,10,5]
        # print('sf = {}, df = {}  , random ={}  ,'.format(sf,ef,(ef-sf-self.input_frame_num+1)))
        assert ef+1-sf == tube_length and tube_length >= self.input_frame_num

        # start_frame_index 从0开始，
        if tube_length == self.input_frame_num:
            start_frame_index = 0
        else:start_frame_index = np.random.randint(start_index,min(self.train_list_skip+start_index,tube_length - self.input_frame_num) , size=1)[0]
        
      

        imgs = []
        img_filelist =[]
        boxes_out = []
        if self.is_train:
            x_flip = np.random.choice([True, False]) # for all frame of a video ,operate flip or crop
            crop = np.random.choice([True, False])
            skip= np.random.choice([1, 2])
            if tube_length - start_frame_index <=32 or not self.is_train: skip =1
        else:
            x_flip = False
            crop = False
            skip = 1
        #print('sf = {}, df = {}, tube_length={},train_start_index={}  , start_frame_index = {} ,end_frame_index = {} ,skip={}'
        #     .format(sf,ef,tube_length,start_index,start_frame_index,(start_frame_index+self.input_frame_num*skip),skip))        
        
        # sample stride
        


        # sample length =16,
        for index in range(start_frame_index+sf,(start_frame_index+sf +self.input_frame_num* skip) ,skip):
            img_name = '{:05d}.jpg'.format(index)
            box_index= index - sf
            #print('box_select[{}] = {}'.format(box_index, box_infos[box_index]))
            
            img_file = os.path.join(self.img_path,str(video_name[:-2]),img_name)
            img_filelist.append(img_file)
            # print('img_file = ' ,img_file)
            img = cv2.imread(img_file) # h,w,c
            img = img[:,:,::-1] # convert BGR to RGB
            img = np.transpose(img,(2,0,1)) # to c,h,w
            # preprocess

            
            img, bbox = self.tsf((img, box_infos[box_index],x_flip,crop,self.is_train))
            # print('bbox[{}] = {}'.format((index - sf), bbox))
            # height, width, _ = img.shape # should be  C, H, W
            # print('img.shape : ' ,img.shape, ' img = ',img[0,0,:])

            imgs.append(img)
            boxes_out.append(bbox)



        imgs = np.asarray(imgs,dtype=np.float32) #  L,C, H, W
        boxes_out = np.asarray(boxes_out,dtype=np.float32) # [帧数,10,5]
        # print('imgs len = ',len(imgs))

        rgb_images = torch.from_numpy(imgs).permute(1,0,2,3) # convert to    C,L, H, W
        # box_select = box_infos[start_frame_index + (self.input_frame_num // 2)]  # select middle frame,.....need ,flip crop
        selected_index= self.input_frame_num // 2
        
        if self.box_select_mode == 1:
            
            
            box_select = boxes_out[selected_index]
        elif self.box_select_mode == 2:
            box_select = np.mean(boxes_out, axis=0,  keepdims=False)
        elif self.box_select_mode == 3:
            a = boxes_out[:,:,0][boxes_out[:,:,0]>0]
            if len(a) % self.input_frame_num != 0:
                print( 'box incorrecs: len(a) = {}'.format(len(a)))
                box_select = boxes_out[selected_index]
            else:
                box_x1 = np.min(boxes_out[:,:,0], axis=0,  keepdims=True)
                box_y1 = np.min(boxes_out[:,:,1], axis=0,  keepdims=True)
                box_x2 = np.max(boxes_out[:,:,2], axis=0,  keepdims=True)
                box_y2 = np.max(boxes_out[:,:,3], axis=0,  keepdims=True)
                box_class= boxes_out[selected_index:selected_index+1,:,4]
                #print(box_x1,box_class)
                box_select = np.transpose(np.concatenate((box_x1,box_y1,box_x2,box_y2,box_class),axis=0)) 
            
        #print('box all = {}, box_select= {},x_flip = {},skip = {}'.format(boxes_out[:,:,4],box_select,selected_index,x_flip,skip)) 
            
            
            
            
        
        #print('boxes len = {},select = {}'.format(len(boxes_out),selected_index))
        boxes = torch.from_numpy(np.array(box_select, dtype=np.float32)) # [10,5]
        num_box = np.array([box_select.shape[0]], dtype=np.long)

        # print('rgb_images shape = {}, boxes shape = {}'.format(rgb_images.size(), boxes.size()))
        if self.is_train:
            return rgb_images,boxes,num_box
        else:
            return rgb_images, boxes, num_box,img_filelist




    def __len__(self):
        return self.video_len