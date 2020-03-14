import numpy as np



def generate_anchors(base_size =16,ratios=[0.5,1,2],scales = 2 ** np.arange(3,6)):
    """
    生成9个anchors ，3种比例，3种大小
    :param base_size:
    :param ratios:
    :param scales:
    :return:
    """
    base_anchor = np.array([1, 1, base_size, base_size]) - 1  # base_anchor = array([ 0,  0, 15, 15])
    # print(base_anchor)
    ratio_anchors = _ratio_enum(base_anchor, ratios)  # 生成3个不同宽高比的矩阵
    a = np.vstack(_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0]))
    # print(a)
    return a



def _whctrs(anchor):
    # 返回一个anchor的宽, 高, 以及中心点的(x,y)坐标值
    w = anchor[2] - anchor[0] + 1 # eg:15-0+1 = 16
    h = anchor[3] - anchor[1] + 1 # eg:15-0+1 = 16
    x_ctr = anchor[0] + 0.5 * (w-1) # eg: 0+0.5*(16-1) = 7.5
    y_ctr = anchor[1] + 0.5 * (h-1) # eg:0+0.5*(16-1) = 7.5

    return  w,h,x_ctr,y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    # 给定一组围绕中心点(x_ctr, y_ctr) 的 widths(ws) 和 heights(hs) 序列, 输出对应的 anchors
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    # anchors里面的坐标分别对应着左上角的坐标和右下角的坐标
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    生成3个不同宽高比的矩阵
    [[-3.5   2.25 18.5  12.75]
     [ 0.    0.   15.   15.  ]
     [ 2.5  -3.   12.5  18.  ]]
    """
    w,h,x_ctr,y_ctr = _whctrs(anchor)# 16,16,7.5,7.5
    # print(w,h,x_ctr,y_ctr) # 16,16,7.5,7.5
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = ws * ratios
    # print(ws,hs) # 生成不同宽高比的值 [23,16,11],[11.5,16,22]
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    # print(anchors)
    return anchors

def _scale_enum(anchor, scales):
    """
    生成3个不同大小的矩阵
    根据给定的anchor(box), 枚举其所有可能scales的anchors(boxes)
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    # print(w)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors




a = generate_anchors()

