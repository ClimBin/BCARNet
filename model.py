import os
import torch
import torch.nn as nn

import torch.nn.functional as F
from backbone import pvt_v2_b2,resnet50
from timm.models.layers import to_2tuple

from pdb import set_trace

def _upsample_like(src,tar):

    src = F.upsample(src, size=tar.shape[2:], mode='bilinear', align_corners=True)
    return src

upx = {"x2":nn.UpsamplingBilinear2d(scale_factor=2), "x4":nn.UpsamplingBilinear2d(scale_factor=4),
                "x8":nn.UpsamplingBilinear2d(scale_factor=8), "x16":nn.UpsamplingBilinear2d(scale_factor=16),
                "x32":nn.UpsamplingBilinear2d(scale_factor=32)}

class DWConv_Mulit(nn.Module):
    def __init__(self, dim=768):
        super(DWConv_Mulit, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv_Mulit(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, g, H, W):
        B, N, C = x.shape
        q = self.q(g).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        print("attn-----q",q.shape)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        print("attn-----k,v",k.shape,v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        print("attn---attn",attn.shape)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        print("attn---x",x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, q, k

class Block(nn.Module):

    def __init__(self, dim, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, g, H, W):
        msa, q, k = self.attn(self.norm1(x),self.norm2(g), H, W)
        print("block--msa,q,k",msa.shape,q.shape,k.shape)
        
        x = x + g + msa
        
        x = x + self.mlp(self.norm2(x), H, W)
        print("block",x.shape)

        return x, q, k
    
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        

    def forward(self, x):
        x = self.proj(x)

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class CrossAttentionModule(nn.Module):
    def __init__(self, in_planes, out_planes, mode = "lh"):
        super(CrossAttentionModule, self).__init__()

        self.patch_embed_norm = OverlapPatchEmbed(img_size=224 // 4, patch_size=3, stride=1, in_chans=in_planes,
                                             embed_dim=out_planes)
        self.patch_embed_down = OverlapPatchEmbed(img_size=224 // 4, patch_size=3, stride=1, in_chans=out_planes,
                                             embed_dim=out_planes)
        self.block = Block(dim=out_planes)
        self.norm = nn.LayerNorm(out_planes)
        self.upx2 = upx['x2']
        self.mode = mode
        #Temporarily hidden as the manuscript is under review

    def forward(self, Fkv, Fq):  # l->h, Fq:down, Fkv,norm   h->l, Fq:norm, Fkv:down 
        B = Fkv.shape[0]
        if self.mode == 'lh':
            #Temporarily hidden as the manuscript is under review
            pass
        else:
            #Temporarily hidden as the manuscript is under review
            pass
        

        #Temporarily hidden as the manuscript is under review
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.sigmoid(y)
        return x * y

class conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class CFM(nn.Module):  #cascaded fusion module
    def __init__(self, ):
        super().__init__()

        self.upx2 = upx['x2']
        self.upx4 = upx['x4']

        self.conv1 = conv2d(32,32,1,0)
        
        self.conv2 = conv2d(64,32,1,0)

        self.conv3_1 = conv2d(64,32,1,0)
        self.conv3_2 = nn.Sequential(
            conv2d(32,32,3,1),
            nn.Dropout2d(0.1,False),
            nn.Conv2d(32,1,3,1,1,bias=False)
        )

    def forward(self, xh, xm, xl):
        xh = self.conv1(xh)
        xh = self.upx2(xh)
        xm_ = xm
        xm  = torch.cat((xh,xm),1)
        xm = self.conv2(xm)
        xm = xm + xm_
        xm = self.upx2(xm)
        xl_ = xl
        xl = torch.cat((xm,xl),1)
        xl = self.conv3_1(xl)
        xl = xl + xl_

        out = self.conv3_2(xl)
        out = self.upx4(out)

        return out

class BiAttention(nn.Module):
    def __init__(self, in_channel):
        super(BiAttention, self).__init__()
        self.conv_h = nn.Linear(in_channel, in_channel)
        self.conv_w = nn.Linear(in_channel, in_channel)
        self.conv = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channel),
                                  nn.ReLU()
                                  )

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        #Temporarily hidden as the manuscript is under review
        pass


class ICAN(nn.Module):
    def __init__(self, args):
        super(ICAN, self).__init__()
        
        # output features channel number of backbone 
        self.channel_num = args.channels   # List, ascending 

        #backbone
        if args.backbone_name == 'resnet':
            self.backbone = resnet101()
            path = 'resnet50.pth'
        elif args.backbone_name == 'pvt':
            self.backbone = pvt_v2_b2()
            path = 'pvt_v2_b2.pth'

        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        #compressed units
        self.compr1_conv = nn.Sequential(
            nn.Conv2d(self.channel_num[0], 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.compr1_ca = ChannelAttention(32)
        self.compr2_conv = nn.Sequential(
            nn.Conv2d(self.channel_num[1], 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.compr2_ca = ChannelAttention(32)
        self.compr3_conv = nn.Sequential(
            nn.Conv2d(self.channel_num[2], 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.compr3_ca = ChannelAttention(32)
        self.compr4_conv = nn.Sequential(
            nn.Conv2d(self.channel_num[3], 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.compr4_ca = ChannelAttention(32)

        self.conv1 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            BiAttention(32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            BiAttention(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            BiAttention(32)
        )
        
        # self.cros_att1_1 = CrossAttentionModule(32,32)      
        # self.cros_att2_1 = CrossAttentionModule(32,32) 
        # self.cros_att3_1 = CrossAttentionModule(32,32)

        self.cros_att1_2 = CrossAttentionModule(32,32, mode='hl')          
        self.cros_att2_2 = CrossAttentionModule(32,32, mode='hl')
        self.cros_att3_2 = CrossAttentionModule(32,32, mode='hl')

        #Temporarily hidden as the manuscript is under review

    def compress_channel(self, block_conv, block_ca, feature):
        #Temporarily hidden as the manuscript is under review
        pass
        
    def forward(self, images):
        # backbone features
        f1, f2, f3, f4 = self.backbone(images)    # low to high

        
        f1  = self.compress_channel(self.compr1_conv, self.compr1_ca, f1)
        f2  = self.compress_channel(self.compr2_conv, self.compr2_ca, f2)
        f3  = self.compress_channel(self.compr3_conv, self.compr3_ca, f3)
        f4  = self.compress_channel(self.compr4_conv, self.compr4_ca, f4)
        
        # cros3_lh = self.cros_att3_1(f4,f3)  
        set_trace()
        cros3_hl = self.cros_att3_2(f3,f4)
        cros3_hl = self.att_conv3(cros3_hl)
        cat3 = torch.cat((cros3_hl,upx['x2'](f4)),dim=1)  # heat map
        cat3_out = self.conv3(cat3)   # heat map
        side3 = self.side3out(cat3)
        side3 = upx['x16'](side3)

        # cros2_lh = self.cros_att2_1(cros3_out,f2)
        cros2_hl = self.cros_att2_2(f2,cat3_out)
        cros2_hl = self.att_conv2(cros2_hl)
        cat2 = torch.cat((cros2_hl,upx['x2'](cat3_out)),dim=1)
        cat2_out = self.conv2(cat2)
        side2 = self.side2out(cat2)
        side2 = upx['x8'](side2)

        # cros1_lh = self.cros_att1_1(cros2_out,f1)
        cros1_hl = self.cros_att1_2(f1,cat2_out)
        cros1_hl = self.att_conv1(cros1_hl)
        cat1 = torch.cat((cros1_hl, upx['x2'](cat2_out)),dim=1)
        cat1_out = self.conv1(cat1)

        side1 = self.side1out(cat1)
        side1 = upx['x4'](side1)

        res = self.final_out(cat1_out)
        res = upx['x4'](res)


        return res, side1, side2, side3



