# 启动命令: CUDA_VISIBLE_DEVICES=2 python trainmyrcnn.py

# 1 lib/datasets/ucf24data: lib/datasets/ucf24data/generate_trainlist_and_anno.py 生成训练列表和数据

# 2 ucf24dataset_skip.py 训练过程中调用的生成batch数据

# 3 trainmyrcnn训练model

# 4 model/ava 存放模型


# 每次的输入是18帧的视频片段，目前输出有定位和行为分类的结果，但是未加入对不同段视频进行tube连接的方法。具体结果如下图所示：


输入： C,L, H, W；[batch, 3, 16, 240, 320]
tensorflow :input shape : HWC
rpn_cls_score size  anchor w  h = torch.Size([1, 60, 15, 20])

gt_boxes:[x1,y1,x2,y2,label],左上角坐标，右下角坐标,label
生成的box 也一样

ANCHOR_RATIOS = [1,2]# first  [0.6,1,1,4,1.75,2.1] = height / width


if gt_boxes 为[0,0,0,0,0]  overlaps会置位0，anchor若为0，overlap置位-1

maybe your x,y coordinates has -1

8个anchor，fill 0 is ok

8个 anchor person，every ball



im_info[0][1] width
im_info[0][0] heifht
im_info[0][2] scale,输入图片与原图片的比例.

输入图片:width 320,height 240  opencv出来是HWC,BGR,输入时CLHW,RGB


!!!!!!! very important,origin anno ,one tube if conrrecd one action,two tube,maybe same.

