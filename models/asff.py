import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from PIL import Image
from torch.cuda import amp

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class ASFFV5(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=True):
        """
        ASFF version for YoloV5 .
        different than YoloV3
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be 
        512, 256, 128 -> multiplier=1
        256, 128, 64 -> multiplier=0.5
        For even smaller, you need change code manually.
        """
        super(ASFFV5, self).__init__()
        self.level = level
        self.dim = [512,512,256,128]
        # print(self.dim)
        
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(512, 512, 3, 2)
                
            self.stride_level_2 = Conv(256, 512, 3, 2)
 
            self.stride_level_3 = Conv(128,512, 3, 2)            
                
            self.expand = Conv(512, 512, 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(512, 512, 1, 1)
 
            self.stride_level_2 = Conv(256, 512, 3, 2)
 
            self.stride_level_3 = Conv(128, 512, 3, 2)
 
            self.expand = Conv(512, 512, 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(512, 256, 1, 1)
 
            self.compress_level_1 = Conv(512, 256, 1, 1)
 
            self.stride_level_3 = Conv(128, 256, 3, 2)
 
            self.expand = Conv(256, 256, 3, 1)
        elif level == 3:
            self.compress_level_0 = Conv(512, 128, 1, 1)
 
            self.compress_level_1 = Conv(512, 128, 1, 1)   
 
            self.compress_level_2 = Conv(256, 128, 1, 1)  
 
            self.expand = Conv(128, 128, 3, 1)
        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            512, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            512, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            256, compress_c, 1, 1)
        self.weight_level_3 = Conv(
            128, compress_c, 1, 1)
        self.weight_levels = Conv(
            compress_c*3, 3, 1, 1)
        self.vis = vis

    def forward(self, x): #l,m,s
        """
        # 128, 256, 512
        512, 256, 128
        from small -> large
        """
        x_level_0=x[3] #l
        x_level_1=x[2] #m
        x_level_2=x[1] #s
        x_level_3=x[0] #ss
        # print('x_level_0: ', x_level_0.shape)
        # print('x_level_1: ', x_level_1.shape)
        # print('x_level_2: ', x_level_2.shape)
        if self.level == 0:
            level_0_resized = x_level_0
 
            level_1_resized = self.stride_level_1(x_level_1)
 
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
 
            level_3_downsampled_inter = F.max_pool2d(x_level_3, 3, stride=4, padding=1)
            level_3_resized = self.stride_level_3(level_3_downsampled_inter)
            
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
 
            level_1_resized = x_level_1
 
            level_2_resized = self.stride_level_2(x_level_2)
 
            level_3_downsampled_inter = F.max_pool2d(x_level_3, 3, stride=2, padding=1)
            level_3_resized = self.stride_level_3(level_3_downsampled_inter)
 
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')   
 
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')
 
            level_2_resized = x_level_2
 
            level_3_resized = self.stride_level_3(x_level_3)
 
        elif self.level == 3:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=8, mode='nearest')  
 
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=4, mode='nearest')   
 
            x_level_2_compressed = self.compress_level_2(x_level_2)
            level_2_resized = F.interpolate(x_level_2_compressed, scale_factor=2, mode='nearest')
 
            level_3_resized = x_level_3
 
 
        print('level: {}, l1_resized: {}, l2_resized: {}, l3_resized: {}'.format(self.level,level_1_resized.shape, level_2_resized.shape,level_3_resized.shape))
 
 
 
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)
 
 
 
 
        print('level_0_weight_v: ', level_0_weight_v.shape)
        print('level_1_weight_v: ', level_1_weight_v.shape)
        print('level_2_weight_v: ', level_2_weight_v.shape)
        print('level_3_weight_v: ', level_3_weight_v.shape)
 
 
 
 
        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
 
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] +\
            level_1_resized * levels_weight[:, 1:2, :, :] +\
            level_2_resized * levels_weight[:, 2:3 :, :] +\
            level_3_resized * levels_weight[:, 3:, :, :]
 
 
        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out 


class ASFFDetect(nn.Module):   #add ASFFV5 layer and Rfb 
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export dummp parameter
    export = False  # export mode

    def __init__(self, nc=1, anchors=(), ch=(), multiplier=0.5,rfb=False,inplace=True):  # detection layer
        super(ASFFDetect,self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.l0_fusion = ASFFV5(level=0, multiplier=multiplier,rfb=rfb)
        self.l1_fusion = ASFFV5(level=1, multiplier=multiplier,rfb=rfb)
        self.l2_fusion = ASFFV5(level=2, multiplier=multiplier,rfb=rfb)
        self.l3_fusion = ASFFV5(level=3, multiplier=multiplier,rfb=rfb)
        #self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        #a = torch.tensor(anchors).float().view(4, 4, )
        #self.register_buffer('anchors', a)  # shape(nl,na,2)
        #self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        #self.inplace = inplace  # use in-place ops (e.g. slice assignment)
 
    def forward(self, x):
        print(self.na)
        z = []  # inference output
        result=[]
       
        result.append(self.l3_fusion(x))        
        result.append(self.l2_fusion(x))
        result.append(self.l1_fusion(x))
        result.append(self.l0_fusion(x))
        x=result      
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
 
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
 
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.tensor_split((2, 4), 4)
                    xy = (xy * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))
 
        return x if self.training else (torch.cat(z, 1), x)
 
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        shape = 1, self.na, ny, nx, 2  # grid shape
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(torch.arange(ny).to(d), torch.arange(nx).to(d), indexing='ij')
        else:
             yv, xv = torch.meshgrid(torch.arange(ny, device=d), torch.arange(nx, device=d))
        grid = torch.stack((xv, yv), 2).expand(shape).float()
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape).float()
        return grid, anchor_grid
        
        
class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
    
    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)            
        else:
            out = (torch.cat(z, 1), x)

        return out
    
    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1,c2,_,_ = self.m[i].weight.shape
            c1_,c2_, _,_ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1,c2),self.ia[i].implicit.reshape(c2_,c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0,1)
            
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)
