from lib.model.ava.i3d_net import I3D
from lib.model.ava.pytorch_i3d import InceptionI3d
import torch
from collections import OrderedDict

model_path = '/home/hyshuai/action_detection/pytorch-i3d/models/rgb_imagenet.pt'
state_dict = torch.load(model_path)
print(type(state_dict))

para_old = OrderedDict()
i3d = I3D()



# i3d.cuda()
# i3d.train()
# print(i3d.state_dict)
for name,param in i3d.state_dict().items():
    name_new = name.replace('features.','').replace('branch0.0','b0').replace('branch1.0','b1a').replace('branch1.1','b1b')\
        .replace('branch2.0','b2a').replace('branch2.1','b2b').replace('branch3.1','b3b').replace('branch2.0','b2a').replace('Logits','logits.conv3d')
    para_old[name] = state_dict[name_new]
    print(name_new)
print("---------------------")
print("---------------------")
print("---------------------")
print("---------------------")
#
# for k , v in para_old.items():
#     print(k)
#     # print(v[0,0,0])
#
print("---------------------loading------------------")
i3d.load_state_dict(para_old)
from torch.autograd import variable

with torch.no_grad():
    a = variable(torch.rand((1,3,64,224,224))).cuda()
    im_info = variable(torch.from_numpy(np.array([[224,224,16]],dtype=np.float32))).cuda()
    num_boxes = variable(torch.from_numpy(np.array([[2]],dtype=np.long))).cuda() # (b,k)
    gt_boxes = variable(torch.from_numpy(np.array([[[10,10,40,40,0],[50,50,150,150,1]]],dtype=np.float32))).cuda()  # (b, K, 5)

    net = i3d_rcnn(classes_num=8)
    net.create_architecture()
    print(net)
    net.cuda()
    net.eval()


    for i in range(100):
        start = time.time()
        _, _, bbox_pred, _, _, _, _, _ = net(a,im_info,gt_boxes,num_boxes)

        print('epoch time = {} bbox_pred = {} '.format((str((time.time() - start) * 1000)),bbox_pred))
        # print('----------------------------------------------')
        # print('----------------------------------------------')
