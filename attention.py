'''
ATTENTION MODULE

'''
import torch
import torch.nn as nn


class AttentionNet(nn.Module):
    '''
    Attention module to extract features for video reconstruction
    Input
        lowres    : low-res video extracted from coded exposure image;            size (N,9,H/3,W/3) 
        blurred   : shuffled video obtained from fully exposed (blurred) image;   size (N,9,H/3,W/3)
    Output
        final_map : combined feature map using attention;   size (N,128*2,H/3,W/3)
        attn_map  : computed attention map;                 size (N,1,H/3,W/3)

    '''
    ## SHARED ENCODER BLOCK
    def encoder_block(self, in_channels, out_channels):
        # input  : (N,9,H/3,W/3)
        # output : (N,128,H/3,W/3)
        layers = []

        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=64, 
                                kernel_size=3, stride=1, padding=1, dilation=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, 
                                kernel_size=3, stride=1, padding=1, dilation=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=out_channels, 
                                kernel_size=3, stride=1, padding=1, dilation=1))
        return nn.Sequential(*layers)

    ## ATTENTION BLOCK
    def attention_block(self, map_lowres, map_blurred):
        # inputs : (N,128,H/3,W/3) (N,128,H/3,W/3)
        # output : (N,1,H/3,W/3)
        mapA = map_lowres / torch.norm(map_lowres, dim=1, keepdim=True)
        mapB = map_blurred / torch.norm(map_blurred, dim=1, keepdim=True)
        mapC = torch.sum(mapA*mapB, dim=1, keepdim=True)
        mapC = (mapC + 1) / 2
        return mapC
   
    def __init__(self, in_channels, out_channels):
        super(AttentionNet, self).__init__()
        self.encoder = self.encoder_block(in_channels=in_channels, out_channels=out_channels)
    
    def forward(self, lowres, blurred):
        # encoder blocks
        map_lowres = self.encoder(lowres)
        map_blurred = self.encoder(blurred)        
        # attention block
        attn_map = self.attention_block(map_lowres, map_blurred)
        final_map = torch.cat([map_lowres*(1-attn_map), map_blurred*attn_map], dim=1)
        return final_map, attn_map