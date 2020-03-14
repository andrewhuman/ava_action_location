from torch.nn.modules.module import Module
from ..functions.roi_crop import RoICropFunction

class _RoICrop(Module):
    def forward(self, input1, input2):
        return RoICropFunction()(input1, input2)
    def __init__(self, layout = 'BHWD'):
        super(_RoICrop, self).__init__()
