import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, ch=32):
        super(Discriminator, self).__init__()

        self.conv1_1 = spectral_norm(nn.Conv2d(3,    ch*1, kernel_size=3, padding=1))
        self.conv1_2 = spectral_norm(nn.Conv2d(ch*1, ch*1, kernel_size=3, stride=2, padding=1))

        self.conv2_1 = spectral_norm(nn.Conv2d(ch*1, ch*2, kernel_size=3, padding=1))
        self.conv2_2 = spectral_norm(nn.Conv2d(ch*2, ch*2, kernel_size=3, stride=2, padding=1))

        self.conv3_1 = spectral_norm(nn.Conv2d(ch*2, ch*4, kernel_size=3, padding=1))
        self.conv3_2 = spectral_norm(nn.Conv2d(ch*4, ch*4, kernel_size=3, stride=2,padding=1))

        self.conv4_1 = spectral_norm(nn.Conv2d(ch*4, ch*8, kernel_size=3, padding=1))
        self.conv4_2 = spectral_norm(nn.Conv2d(ch*8, ch*8, kernel_size=3, stride=2,padding=1))

        self.conv5_1 = spectral_norm(nn.Conv2d(ch*8, ch*16, kernel_size=3, padding=1))
        self.conv5_2 = spectral_norm(nn.Conv2d(ch*16,ch*16, kernel_size=3, stride=2,padding=1))

        self.conv6_1 = spectral_norm(nn.Conv2d(ch*16, 1, kernel_size=3, padding=1))

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self._weight_initialize()

    def _weight_initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.act(x)
        x = self.conv1_2(x)
        x = self.act(x)

        x = self.conv2_1(x)
        x = self.act(x)
        x = self.conv2_2(x)
        x = self.act(x)

        x = self.conv3_1(x)
        x = self.act(x)
        x = self.conv3_2(x)
        x = self.act(x)

        x = self.conv4_1(x)
        x = self.act(x)
        x = self.conv4_2(x)
        x = self.act(x)

        x = self.conv5_1(x)
        x = self.act(x)
        x = self.conv5_2(x)
        x = self.act(x)

        x = self.conv6_1(x)
        return x
    
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

'''预处理模块''' 
class Pre_block(nn.Module):   
    def __init__(self, in_channels, out_channels):
        super(Pre_block, self).__init__()
        self.conv1 = ConvLayer(in_channels, 16, kernel_size=11, stride=1) 
        self.conv2 = nn.Sequential(nn.Conv2d(16, out_channels, kernel_size=3, stride=2, padding=1),
#                                   nn.PReLU() 
                                    )
        
    def forward(self, x): 
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2

'''后处理模块'''
class Post_block(nn.Module):   
    def __init__(self, in_channels, out_channels):
        super(Post_block, self).__init__()
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
#                                   nn.PReLU() 
                                    )
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0)  

    def forward(self, x): 
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2

'''密集残差模块'''
class RD_block(nn.Module):   
    def __init__(self, in_channels, out_channels, growth_rate):
        super(RD_block, self).__init__()
        inner_channel = 4*growth_rate
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.PReLU() )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels+(1*growth_rate), inner_channel, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.PReLU() )  
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels+(2*growth_rate), inner_channel, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.PReLU() ) 
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels+(3*growth_rate), inner_channel, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.PReLU() )  
        self.conv_fine = nn.Conv2d(in_channels+(4*growth_rate), out_channels, kernel_size=1, stride=1, padding=0)          
        
    def forward(self, x): 
        y1 = self.conv1(x)
        x1 = torch.cat([x, y1], 1) 
        y2 = self.conv2(x1)
        x2 = torch.cat([x1, y2], 1)
        y3 = self.conv3(x2)
        x3 = torch.cat([x2, y3], 1)
        y4 = self.conv4(x3)
        x4 = torch.cat([x3, y4], 1)
        y5 = self.conv_fine(x4)
        return y5


class RD_group(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate):
        super(RD_group, self).__init__()
        
        self.R1  = RD_block(in_channels, out_channels, growth_rate)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.R2  = RD_block(in_channels, out_channels, growth_rate)
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.R3  = RD_block(in_channels, out_channels, growth_rate)
        self.gamma3 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        x1  = self.R1(x)  + self.gamma1*residual
        x2  = self.R2(x1) + self.gamma2*residual
        x3  = self.R3(x2) + self.gamma3*residual
        return x3 


'''扩张残差模块'''
class RI_block(nn.Module):   
    def __init__(self, in_channels, out_channels):
        super(RI_block, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, dilation=2, padding=2)
        self.conv1_3 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, dilation=3, padding=3)
        self.conv1_4 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, dilation=4, padding=4)
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0),
            nn.PReLU() )  
        
        self.conv2_1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, dilation=2, padding=2)
        self.conv2_3 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, dilation=3, padding=3)
        self.conv2_4 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, dilation=4, padding=4)
        self.conv2_5 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0)            
        
    def forward(self, x): 
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)
        x4 = self.conv1_4(x)
        y  = self.conv1_5(torch.cat([x1, x2, x3, x4], 1))
        
        y1 = self.conv2_1(y)
        y2 = self.conv2_2(y)
        y3 = self.conv2_3(y)
        y4 = self.conv2_4(y)
        z  = self.conv2_5(torch.cat([y1, y2, y3, y4], 1))
        return z + x


'''SE注意力模块''' 
class SE_Attn(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Attn, self).__init__() 
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.demo = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(channel//reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid() )

    def forward(self, x):
        w = self.demo(self.avg_pool(x))
        return w
    
    
'''SA注意力模块''' 
class SA_Attn(nn.Module):
    def __init__(self, channel):
        super(SA_Attn, self).__init__()
    
        self.demo = nn.Sequential(
            nn.Conv2d(channel, channel//8, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(channel//8, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
        
    def forward(self, x):
        w = self.demo(x)
        return w 
    
    
'''双注意力模块''' 
class Dual_Attn(nn.Module):
    def __init__(self, channel):
        super(Dual_Attn, self).__init__()
    
        self.space = SA_Attn(channel)
        self.chanl = SE_Attn(channel)
        
    def forward(self, x):
        s_weight = self.space(x)
        c_weight = self.chanl(x)
        atts = x * s_weight
        attc = atts * c_weight
        return attc + x  

'''Self注意力模块''' 
class Self_Attn(nn.Module):
    def __init__(self, in_ch):
        super(Self_Attn, self).__init__()   
        med_ch = in_ch//2
        self.conv_o  = nn.Conv2d(in_ch, med_ch, kernel_size=1, stride=1, padding=0)
        self.conv_v  = nn.Conv2d(in_ch, med_ch, kernel_size=1, stride=1, padding=0)
        self.conv_s  = nn.Conv2d(in_ch, med_ch, kernel_size=1, stride=1, padding=0)
        self.conv    = nn.Conv2d(med_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.relu    = nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batchsize, channel, width, height = x.size()
        x_o = self.conv_o(x).view(batchsize, -1, width*height).permute(0,2,1)
        x_v = self.conv_v(x).view(batchsize, -1, width*height)
        x_v = self.softmax(x_v)
        x_s = self.conv_v(x).view(batchsize, -1, width*height).permute(0,2,1) 
        x_s = self.softmax(x_s)

        A = torch.bmm(x_v, x_s)
        B = torch.bmm(x_o, A).permute(0,2,1)
        C = B.view(batchsize, -1, width, height)
        out = self.relu(self.conv(C)) 
        result = self.gamma*out + x
        return result
    
#class Self_Attn(nn.Module):
#    def __init__(self, in_dim):
#        super(Self_Attn, self).__init__()
#        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, stride=1, padding=0)
#        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, stride=1, padding=0)
#        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1, padding=0)
#        self.softmax    = nn.Softmax(dim=-1) 
#        self.gamma = nn.Parameter(torch.zeros(1))
#        
#    def forward(self, x):
#        b, c, w, h = x.size()
##        proj_value  = self.value_conv(x).view(b, -1, w*h) # b c hw
##        proj_query  = self.query_conv(x).view(b, -1, w*h) # b wh c
##        proj_key    = self.key_conv(x).view(b, -1, w*h)   # b c hw
##        attention   = self.softmax(torch.bmm(proj_query.permute(0,2,1), proj_key))
##        result = torch.bmm(proj_value, attention.permute(0,2,1) )
##        result = result.view(b, c, w, h)
#        proj_value  = self.value_conv(x).view(b, -1, w*h) # b c wh
#        proj_query  = self.query_conv(x).view(b, -1, w*h) # b hw c/8
#        proj_key    = self.key_conv(x).view(b, -1, w*h)   # b c/8 hw
#        attention   = self.softmax(torch.bmm(proj_value, proj_query.permute(0,2,1))) # b c c/8
#        result = torch.bmm(attention, proj_key) # b c wh
#        result = result.view(b, c, w, h)
#        result = self.gamma * result
#        return result + x
    
 
'''多尺度注意力融合模块'''
'''编码器注意力模块：参数1为注意力模块的输出通道数，参数2-5为之前模块通道数（由前到后）'''
class AttFuse_1(nn.Module):
    def __init__(self, channel, channel_1):
        super(AttFuse_1, self).__init__()   
        
        self.fea_1_1 = Dual_Attn(channel_1)
        self.fea_1_2 = nn.Sequential(nn.Conv2d(channel_1, channel_1,   kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(channel_1, channel_1*2, kernel_size=1, stride=1, padding=0))
        self.ref = nn.Sequential(
            nn.Conv2d(channel_1*2, channel, kernel_size=1, stride=1, padding=0),
            nn.PReLU() )
        self.SATT = Self_Attn(channel)
        
    def forward(self, a):
        A = self.fea_1_2(self.fea_1_1(a))
        fuse = self.ref(A)
        out  = self.SATT(fuse)
        return out  


class AttFuse_2(nn.Module):
    def __init__(self, channel, channel_1, channel_2):
        super(AttFuse_2, self).__init__()   
        
        self.fea_1_1 = Dual_Attn(channel_1)
        self.fea_1_2 = nn.Sequential(nn.Conv2d(channel_1, channel_1,   kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(channel_1, channel_1*2, kernel_size=1, stride=1, padding=0))
        self.fea_2_1 = Dual_Attn(channel_2)
        self.fea_2_2 = nn.Sequential(nn.Conv2d(channel_2, channel_2,   kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(channel_2, channel_2*2, kernel_size=1, stride=1, padding=0))   
        self.ref = nn.Sequential(
            nn.Conv2d(channel_2*2, channel, kernel_size=1, stride=1, padding=0),
            nn.PReLU() )
        self.SATT = Self_Attn(channel)
        
    def forward(self, a, b):   
        A = self.fea_1_2(self.fea_1_1(a))
        B = self.fea_2_2(self.fea_2_1(b)+A)
        fuse = self.ref(B)
        out  = self.SATT(fuse)
        return out 
    
    
class AttFuse_3(nn.Module):
    def __init__(self, channel, channel_1, channel_2, channel_3):
        super(AttFuse_3, self).__init__()   
        
        self.fea_1_1 = Dual_Attn(channel_1)
        self.fea_1_2 = nn.Sequential(nn.Conv2d(channel_1, channel_1,   kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(channel_1, channel_1*2, kernel_size=1, stride=1, padding=0))
        self.fea_2_1 = Dual_Attn(channel_2)
        self.fea_2_2 = nn.Sequential(nn.Conv2d(channel_2, channel_2,   kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(channel_2, channel_2*2, kernel_size=1, stride=1, padding=0))  
        self.fea_3_1 = Dual_Attn(channel_3)
        self.fea_3_2 = nn.Sequential(nn.Conv2d(channel_3, channel_3,   kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(channel_3, channel_3*2, kernel_size=1, stride=1, padding=0))
        self.ref = nn.Sequential(
            nn.Conv2d(channel_3*2, channel, kernel_size=1, stride=1, padding=0),
            nn.PReLU() )
        self.SATT = Self_Attn(channel)
        
    def forward(self, a, b, c):    
        A = self.fea_1_2(self.fea_1_1(a))
        B = self.fea_2_2(self.fea_2_1(b)+A)
        C = self.fea_3_2(self.fea_3_1(c)+B)
        fuse = self.ref(C)
        out  = self.SATT(fuse)
        return out 


class AttFuse_4(nn.Module):
    def __init__(self, channel, channel_1, channel_2, channel_3, channel_4):
        super(AttFuse_4, self).__init__()   
        
        self.fea_1_1 = Dual_Attn(channel_1)
        self.fea_1_2 = nn.Sequential(nn.Conv2d(channel_1, channel_1,   kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(channel_1, channel_1*2, kernel_size=1, stride=1, padding=0))
        self.fea_2_1 = Dual_Attn(channel_2)
        self.fea_2_2 = nn.Sequential(nn.Conv2d(channel_2, channel_2,   kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(channel_2, channel_2*2, kernel_size=1, stride=1, padding=0))  
        self.fea_3_1 = Dual_Attn(channel_3)
        self.fea_3_2 = nn.Sequential(nn.Conv2d(channel_3, channel_3,   kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(channel_3, channel_3*2, kernel_size=1, stride=1, padding=0))
        self.fea_4_1 = Dual_Attn(channel_4)
        self.fea_4_2 = nn.Sequential(nn.Conv2d(channel_4, channel_4,   kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(channel_4, channel_4*2, kernel_size=1, stride=1, padding=0)) 
        self.ref = nn.Sequential(
            nn.Conv2d(channel_4*2, channel, kernel_size=1, stride=1, padding=0),
            nn.PReLU() )
        self.SATT = Self_Attn(channel)
        
    def forward(self, a, b, c, d):    
        A = self.fea_1_2(self.fea_1_1(a))
        B = self.fea_2_2(self.fea_2_1(b)+A)
        C = self.fea_3_2(self.fea_3_1(c)+B)
        D = self.fea_4_2(self.fea_4_1(d)+C)
        fuse = self.ref(D)
        out  = self.SATT(fuse)
        return out 


'''解码器注意力模块：参数1为注意力模块的输出通道数，参数2-5为之前模块通道数（由前到后），参数6为编码器对应模块通道数'''
class AttFuse_5(nn.Module):
    def __init__(self, channel, channel_1, channel_e): 
        super(AttFuse_5, self).__init__()   
        
        self.fea_1_1 = Dual_Attn(channel_1)
        self.fea_1_2 = nn.ConvTranspose2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.encoder = Dual_Attn(channel_e)
        self.ref = nn.Sequential(
            nn.Conv2d(channel_1, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU() )
        self.SATT = Self_Attn(channel)
        
    def forward(self, a, e): 
        A = self.fea_1_2(self.fea_1_1(a))
        fuse = self.ref(self.encoder(e)+A)
        out  = self.SATT(fuse)
        return out 


class AttFuse_6(nn.Module):
    def __init__(self, channel, channel_1, channel_2, channel_e): 
        super(AttFuse_6, self).__init__()   
        
        self.fea_1_1 = Dual_Attn(channel_1)
        self.fea_1_2 = nn.ConvTranspose2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.fea_2_1 = Dual_Attn(channel_2)
        self.fea_2_2 = nn.ConvTranspose2d(channel_2, channel_2//2, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.encoder = Dual_Attn(channel_e)
        self.ref = nn.Sequential(
            nn.Conv2d(channel_2//2, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU() )
        self.SATT = Self_Attn(channel)
        
    def forward(self, a, b, e):
        A = self.fea_1_2(self.fea_1_1(a))
        B = self.fea_2_2(self.fea_2_1(b)+A)
        fuse = self.ref(self.encoder(e)+B)
        out  = self.SATT(fuse)
        return out 


class AttFuse_7(nn.Module):
    def __init__(self, channel, channel_1, channel_2, channel_3, channel_e): 
        super(AttFuse_7, self).__init__()   
        
        self.fea_1_1 = Dual_Attn(channel_1)
        self.fea_1_2 = nn.ConvTranspose2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.fea_2_1 = Dual_Attn(channel_2)
        self.fea_2_2 = nn.ConvTranspose2d(channel_2, channel_2//2, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.fea_3_1 = Dual_Attn(channel_3)
        self.fea_3_2 = nn.ConvTranspose2d(channel_3, channel_3//2, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.encoder = Dual_Attn(channel_e)
        self.ref = nn.Sequential(
            nn.Conv2d(channel_3//2, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU() )
        self.SATT = Self_Attn(channel)
        
    def forward(self, a, b, c, e):    
        A = self.fea_1_2(self.fea_1_1(a))
        B = self.fea_2_2(self.fea_2_1(b)+A)
        C = self.fea_3_2(self.fea_3_1(c)+B)
        fuse = self.ref(self.encoder(e)+C)
        out  = self.SATT(fuse)
        return out 
    
    
class AttFuse_8(nn.Module):
    def __init__(self, channel, channel_1, channel_2, channel_3, channel_4, channel_e): 
        super(AttFuse_8, self).__init__()   
        
        self.fea_1_1 = Dual_Attn(channel_1)
        self.fea_1_2 = nn.ConvTranspose2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.fea_2_1 = Dual_Attn(channel_2)
        self.fea_2_2 = nn.ConvTranspose2d(channel_2, channel_2//2, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.fea_3_1 = Dual_Attn(channel_3)
        self.fea_3_2 = nn.ConvTranspose2d(channel_3, channel_3//2, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.fea_4_1 = Dual_Attn(channel_4)
        self.fea_4_2 = nn.ConvTranspose2d(channel_4, channel_4//2, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.encoder = Dual_Attn(channel_e)
        self.ref = nn.Sequential(
            nn.Conv2d(channel_4//2, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU() )
        self.SATT = Self_Attn(channel)
        
    def forward(self, a, b, c, d, e):    
        A = self.fea_1_2(self.fea_1_1(a))
        B = self.fea_2_2(self.fea_2_1(b)+A)
        C = self.fea_3_2(self.fea_3_1(c)+B)
        D = self.fea_4_2(self.fea_4_1(d)+C)
        fuse = self.ref(self.encoder(e)+D)
        out  = self.SATT(fuse)
        return out 
   

'''''''''生成器'''''''''    
class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(Generator, self).__init__()
        
        '''颜色编码器'''
        self.Pre_Color    = Pre_block(in_ch, 32)
        self.Ecoder_RDB_1 = RD_group(in_channels=64,  out_channels=64,  growth_rate=32)
        self.Ecoder_RDB_2 = RD_group(in_channels=128, out_channels=128, growth_rate=32)
        self.Ecoder_RDB_3 = RD_group(in_channels=256, out_channels=256, growth_rate=32)
        self.Ecoder_AFM_1 = AttFuse_1(channel=64,  channel_1=32)
        self.Ecoder_AFM_2 = AttFuse_2(channel=128, channel_1=32, channel_2=64)
        self.Ecoder_AFM_3 = AttFuse_3(channel=256, channel_1=32, channel_2=64, channel_3=128)
        self.Ecoder_AFM_4 = AttFuse_4(channel=256, channel_1=32, channel_2=64, channel_3=128, channel_4=256)
        
        '''颜色解码器'''
        self.Decoder_RDB_4 = RD_group(in_channels=256, out_channels=256, growth_rate=32)
        self.Decoder_RDB_5 = RD_group(in_channels=256, out_channels=256, growth_rate=32)
        self.Decoder_RDB_6 = RD_group(in_channels=128, out_channels=128, growth_rate=32)
        self.Decoder_RDB_7 = RD_group(in_channels=64,  out_channels=64,  growth_rate=32)
        self.Decoder_AFM_5 = AttFuse_5(channel=256, channel_1=256, channel_e=256)
        self.Decoder_AFM_6 = AttFuse_6(channel=128, channel_1=256, channel_2=256, channel_e=128)
        self.Decoder_AFM_7 = AttFuse_7(channel=64,  channel_1=256, channel_2=256, channel_3=128, channel_e=64)
        self.Decoder_AFM_8 = AttFuse_8(channel=32,  channel_1=256, channel_2=256, channel_3=128, channel_4=64, channel_e=32)
        self.Post_Color    = Post_block(32, 3)
        
    def forward(self, haze, clean):
        
        '''颜色分支'''
        X  = self.Pre_Color(haze)
        A1 = self.Ecoder_AFM_1(X)
        X1 = self.Ecoder_RDB_1(A1)
        A2 = self.Ecoder_AFM_2(X, X1)
        X2 = self.Ecoder_RDB_2(A2)
        A3 = self.Ecoder_AFM_3(X, X1, X2)
        X3 = self.Ecoder_RDB_3(A3)
        A4 = self.Ecoder_AFM_4(X, X1, X2, X3)
        
        X4 = self.Decoder_RDB_4(A4)
        A5 = self.Decoder_AFM_5(X4, X3)
        X5 = self.Decoder_RDB_5(A5)
        A6 = self.Decoder_AFM_6(X4, X5, X2)
        X6 = self.Decoder_RDB_6(A6)
        A7 = self.Decoder_AFM_7(X4, X5, X6, X1)
        X7 = self.Decoder_RDB_7(A7)
        A8 = self.Decoder_AFM_8(X4, X5, X6, X7, X)
        N = self.Post_Color(A8)

        fakehaze = clean - N
        return fakehaze
       
  