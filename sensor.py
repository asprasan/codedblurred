'''
C2B SENSOR

'''
import torch
import torch.nn as nn
from utils import tensor_type


class C2B(nn.Module):   
    '''
    C2B module to generate coded exposure image and fully exposed (blurred) image 
    from a sequence of sharp frames
    Input
        x       : sequence of sharp frames;         size (N,9,H,W)
    Output
        b1      : coded exposure image;             size (N,1,H,W)
        b0      : complement coded image;           size (N,1,H,W)
        blurred : fully exposed (blurred) image;    size (N,1,H,W)
    Note:
    b1*code_sum + b0*code_comp_sum = blurred*num_frames
    Given two of [b1, b0, blurred] the other can be obtained

    '''
    def __init__(self, code_size=3):
        super(C2B, self).__init__()

        self.code_size = code_size
        # sequential impulse code
        self.code = torch.eye(code_size**2).type(tensor_type)
        self.code = self.code.view(1, code_size**2, code_size, code_size)        
        self.code = nn.Parameter(self.code, requires_grad=False)

    def forward(self, x):
        _, _, H, W = x.size()
        assert H % self.code_size == 0
        assert W % self.code_size == 0
        code_repeat = self.code.repeat(1, 1, H//self.code_size, W//self.code_size)
        code_repeat_comp = 1 - code_repeat
        
        b1 = torch.sum(code_repeat*x, dim=1, keepdim=True) \
                    / torch.sum(code_repeat, dim=1, keepdim=True)
        b0 = torch.sum(code_repeat_comp*x, dim=1, keepdim=True) \
                    / torch.sum(code_repeat_comp, dim=1, keepdim=True)
        blurred = torch.mean(x, dim=1, keepdim=True)
        return b1, b0, blurred
