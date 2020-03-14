# CUDA_VISIBLE_DEVICES=2 python trainmyrcnn.py

# 1 lib/datasets/ucf24data:
#1 lib/datasets/ucf24data/generate_trainlist_and_anno.py 生成训练列表和数据
#2 ucf24dataset_skip.py 训练过程中调用的生成batch数据
#3 trainmyrcnn是训练
#4 model/ava是模型


# 每次的输入是18帧的视频片段，目前输出有定位和行为分类的结果，但是未加入对不同段视频进行tube连接的方法。具体结果如下图所示：