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
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1) 
        
    def forward(self, x): 
        x1 = self.conv(x)
        return x1


'''后处理模块'''
class Post_block(nn.Module):   
    def __init__(self, in_channels, out_channels):
        super(Post_block, self).__init__()
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1)  

    def forward(self, x): 
        x1 = self.conv(x)
        return x1


'''密集残差模块'''
class RD_block(nn.Module):   
    def __init__(self, channel, growth_rate=32):
        super(RD_block, self).__init__()
        inner_channel = 4*growth_rate
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.PReLU() )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel+(1*growth_rate), inner_channel, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.PReLU() )  
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel+(2*growth_rate), inner_channel, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.PReLU() ) 
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel+(3*growth_rate), inner_channel, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.PReLU() )  
        self.conv_fine = nn.Conv2d(channel+(4*growth_rate), channel, kernel_size=3, stride=1, padding=1)          
        
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
        return y5 + x


class RD_group(nn.Module):
    def __init__(self, channel):
        super(RD_group, self).__init__()
        
        self.R1  = RD_block(channel)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.R2  = RD_block(channel)
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.R3  = RD_block(channel)
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.R4  = RD_block(channel)
        self.gamma4 = nn.Parameter(torch.zeros(1))
        self.R5  = RD_block(channel)
        self.gamma5 = nn.Parameter(torch.zeros(1))
        self.R6  = RD_block(channel)
        self.gamma6 = nn.Parameter(torch.zeros(1))
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        residual = x
        x1 = self.R1(x)  + self.gamma1*residual
        x2 = self.R2(x1) + self.gamma2*residual
        x3 = self.R3(x2) + self.gamma3*residual
        x4 = self.R4(x3) + self.gamma4*residual
        x5 = self.R5(x4) + self.gamma5*residual
        x6 = self.R6(x5) + self.gamma6*residual
        y  = self.conv(x6)
        return y


#'''扩张残差模块'''
#class RI_block(nn.Module):   
#    def __init__(self, channel):
#        super(RI_block, self).__init__()
#        self.conv1_1 = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, dilation=1, padding=1)
#        self.conv1_2 = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, dilation=2, padding=2)
#        self.conv1_3 = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, dilation=3, padding=3)
#        self.conv1_4 = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, dilation=4, padding=4)
#        self.conv1_5 = nn.Sequential(
#            nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0),
#            nn.ReLU(inplace=True) )  
#        
#        self.conv2_1 = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, dilation=1, padding=1)
#        self.conv2_2 = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, dilation=2, padding=2)
#        self.conv2_3 = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, dilation=3, padding=3)
#        self.conv2_4 = nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, dilation=4, padding=4)
#        self.conv2_5 = nn.Conv2d(channel*2, channel,  kernel_size=1, stride=1, padding=0)            
#        
#    def forward(self, x): 
#        x1 = self.conv1_1(x)
#        x2 = self.conv1_2(x)
#        x3 = self.conv1_3(x)
#        x4 = self.conv1_4(x)
#        y  = self.conv1_5(torch.cat([x1, x2, x3, x4], 1))
#        
#        y1 = self.conv2_1(y)
#        y2 = self.conv2_2(y)
#        y3 = self.conv2_3(y)
#        y4 = self.conv2_4(y)
#        z  = self.conv2_5(torch.cat([y1, y2, y3, y4], 1))
#        return z + x


#'''SE注意力模块''' 
#class SE_Attn(nn.Module):
#    def __init__(self, channel, reduction=16):
#        super(SE_Attn, self).__init__() 
#        
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        self.demo = nn.Sequential(
#            nn.Conv2d(channel, channel//reduction, kernel_size=1, stride=1, padding=0),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(channel//reduction, channel, kernel_size=1, stride=1, padding=0),
#            nn.Sigmoid() )
#
#    def forward(self, x):
#        w = self.demo(self.avg_pool(x))
#        return w
#    
#    
#'''SA注意力模块''' 
#class SA_Attn(nn.Module):
#    def __init__(self, channel):
#        super(SA_Attn, self).__init__()
#    
#        self.demo = nn.Sequential(
#            nn.Conv2d(channel, channel//8, kernel_size=3, stride=1, padding=1),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(channel//8, 1, kernel_size=1, stride=1, padding=0),
#            nn.Sigmoid())
#        
#    def forward(self, x):
#        w = self.demo(x)
#        return w 
    
    
'''双注意力模块''' 
#class Dual_Attn(nn.Module):
#    def __init__(self, channel):
#        super(Dual_Attn, self).__init__()
#    
#        self.space = SA_Attn(channel)
#        self.chanl = SE_Attn(channel)
#        
#    def forward(self, x):
#        s_weight = self.space(x)
#        c_weight = self.chanl(x)
#        a = x * s_weight
#        b = a * c_weight
#        return b + x  


'''Self注意力模块'''   
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


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class Spatial_Att(nn.Module):
    def __init__(self):
        super(Spatial_Att, self).__init__()
        self.zpool = ZPool()
        self.conv  = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, x):
        x1 = self.zpool(x)
        x2 = self.conv(x1)
        weight = self.sigmoid(x2) 
        return weight
    
class Channel_Att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Att, self).__init__() 
        self.apool = nn.AdaptiveAvgPool2d(1)
        self.demo = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid() )

    def forward(self, x):
        x1 = self.apool(x)
        weight = self.demo(x1)
        return weight
    

class Rotate_Att(nn.Module):
    def __init__(self):
        super(Rotate_Att, self).__init__()
        self.cw = Spatial_Att()
        self.hc = Spatial_Att()
        self.hw = Spatial_Att()
#        self.gamma1 = nn.Parameter(torch.zeros(1))
#        self.gamma2 = nn.Parameter(torch.zeros(1))
#        self.gamma3 = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        x_out = self.hw(x) * x
        
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1) * x_perm1
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()

        x_out = 1/3 * (x_out + x_out11 + x_out21)
#        x_out = self.gamma1*x_out + self.gamma2*x_out11 + self.gamma3*x_out21
        return x_out 


class Cross_Attn(nn.Module):
    def __init__(self, channel):
        super(Cross_Attn, self).__init__()
        self.sha  = nn.Conv2d(in_channels=channel, out_channels=channel//8, kernel_size=1, stride=1, padding=0)
        self.now  = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.satt = Spatial_Att()
        self.catt = Channel_Att(channel)

    def forward(self, pre, now):
        pre_s = self.sha(pre)
        now_s = self.sha(now)
        now_w = self.now(now) 
        fuse = torch.cat([now_s, pre_s], 1)
        w1 = self.satt(fuse) * now_w
        result1 = pre * w1
        w2 = self.catt(result1)
        result2 = result1 * w2
        result = result2 + pre
        return result 


'''多尺度注意力融合模块'''
'''编码器注意力模块：参数1为注意力模块的输出通道数，参数2-5为之前模块通道数（由前到后）'''
class AttFuse_1(nn.Module):
    def __init__(self, channel, channel_1):
        super(AttFuse_1, self).__init__()   
        
        self.fea_down = nn.Sequential(nn.Conv2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1),
                                      nn.Conv2d(channel_1, channel, kernel_size=1, stride=1, padding=0),
                                      )
        self.relu = nn.PReLU()
#        self.SATT = Self_Attn(channel)
        
    def forward(self, a):
        fuse = self.relu(self.fea_down(a))
#        out  = self.SATT(fuse)
        return fuse  


class AttFuse_2(nn.Module):
    def __init__(self, channel, channel_1, channel_2):
        super(AttFuse_2, self).__init__()   
        
        self.fea_A = Cross_Attn(channel_2)
        self.fea_a = nn.Sequential(nn.Conv2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1),
                                   nn.Conv2d(channel_1, channel_2, kernel_size=1, stride=1, padding=0))
        self.fea_down = nn.Sequential(nn.Conv2d(channel_2, channel_2, kernel_size=3, stride=2, padding=1),
                                      nn.Conv2d(channel_2, channel, kernel_size=1, stride=1, padding=0))   
        self.relu = nn.PReLU()
#        self.SATT = Self_Attn(channel)
        
    def forward(self, a, b):   
        A = self.fea_A(self.fea_a(a), b)
        fuse = self.relu(self.fea_down(A))
#        out  = self.SATT(fuse)
        return fuse 
    
    
class AttFuse_3(nn.Module):
    def __init__(self, channel, channel_1, channel_2, channel_3):
        super(AttFuse_3, self).__init__()   
        
        self.fea_A = Cross_Attn(channel_2)
        self.fea_a = nn.Sequential(nn.Conv2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1),
                                    nn.Conv2d(channel_1, channel_2, kernel_size=1, stride=1, padding=0))
        self.fea_B = Cross_Attn(channel_3)
        self.fea_b = nn.Sequential(nn.Conv2d(channel_2, channel_2, kernel_size=3, stride=2, padding=1),
                                    nn.Conv2d(channel_2, channel_3, kernel_size=1, stride=1, padding=0))
        self.fea_down = nn.Sequential(nn.Conv2d(channel_3, channel_3, kernel_size=3, stride=2, padding=1),
                                      nn.Conv2d(channel_3, channel, kernel_size=1, stride=1, padding=0))   
        self.relu = nn.PReLU()
#        self.SATT = Self_Attn(channel)
        
    def forward(self, a, b, c):   
        A = self.fea_A(self.fea_a(a), b) + b
        B = self.fea_B(self.fea_b(A), c) + c
        fuse = self.relu(self.fea_down(B))
#        out  = self.SATT(fuse)
        return fuse 
    


class AttFuse_4(nn.Module):
    def __init__(self, channel, channel_1, channel_2, channel_3, channel_4):
        super(AttFuse_4, self).__init__()   
        
        self.fea_A = Cross_Attn(channel_2)
        self.fea_a = nn.Sequential(nn.Conv2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1),
                                    nn.Conv2d(channel_1, channel_2, kernel_size=1, stride=1, padding=0))
        self.fea_B = Cross_Attn(channel_3)
        self.fea_b = nn.Sequential(nn.Conv2d(channel_2, channel_2, kernel_size=3, stride=2, padding=1),
                                    nn.Conv2d(channel_2, channel_3, kernel_size=1, stride=1, padding=0))
        self.fea_C = Cross_Attn(channel_4)
        self.fea_c = nn.Sequential(nn.Conv2d(channel_3, channel_3, kernel_size=3, stride=2, padding=1),
                                    nn.Conv2d(channel_3, channel_4, kernel_size=1, stride=1, padding=0))
        self.fea_down = nn.Sequential(nn.Conv2d(channel_4, channel_4, kernel_size=3, stride=2, padding=1),
                                      nn.Conv2d(channel_4, channel, kernel_size=1, stride=1, padding=0))   
        self.relu = nn.PReLU()
#        self.SATT = Self_Attn(channel)
        
    def forward(self, a, b, c, d):   
        A = self.fea_A(self.fea_a(a), b) + b
        B = self.fea_B(self.fea_b(A), c) + c
        C = self.fea_C(self.fea_c(B), d) + d
        fuse = self.relu(self.fea_down(C))
#        out  = self.SATT(fuse)
        return fuse 


'''解码器注意力模块：参数1为注意力模块的输出通道数，参数2-5为之前模块通道数（由前到后），参数6为编码器对应模块通道数'''
class AttFuse_5(nn.Module):
    def __init__(self, channel, channel_1, channel_e): 
        super(AttFuse_5, self).__init__()   
        
        self.fea_A = Cross_Attn(channel_e)
        self.fea_a = nn.Sequential(nn.ConvTranspose2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1, output_padding=1),
                                   nn.Conv2d(channel_1, channel_e, kernel_size=1, stride=1, padding=0))
        self.relu = nn.PReLU()
#        self.SATT = Self_Attn(channel)
        
    def forward(self, a, e): 
        A = self.fea_A(self.fea_a(a), e) + e
        fuse = self.relu(A)
#        out  = self.SATT(fuse)
        return fuse 


class AttFuse_6(nn.Module):
    def __init__(self, channel, channel_1, channel_2, channel_e): 
        super(AttFuse_6, self).__init__()   
        
        self.fea_A = Cross_Attn(channel_2)
        self.fea_a = nn.Sequential(nn.ConvTranspose2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1, output_padding=1),
                                   nn.Conv2d(channel_1, channel_2, kernel_size=1, stride=1, padding=0))
        self.fea_B = Cross_Attn(channel_e)
        self.fea_b = nn.Sequential(nn.ConvTranspose2d(channel_2, channel_2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                   nn.Conv2d(channel_2, channel_e, kernel_size=1, stride=1, padding=0))
        self.relu = nn.PReLU()
#        self.SATT = Self_Attn(channel)
        
    def forward(self, a, b, e):
        A = self.fea_A(self.fea_a(a), b) + b
        B = self.fea_B(self.fea_b(A), e) + e
        fuse = self.relu(B)
#        out  = self.SATT(fuse)
        return fuse 


class AttFuse_7(nn.Module):
    def __init__(self, channel, channel_1, channel_2, channel_3, channel_e): 
        super(AttFuse_7, self).__init__()   
        
        self.fea_A = Cross_Attn(channel_2)
        self.fea_a = nn.Sequential(nn.ConvTranspose2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1, output_padding=1),
                                   nn.Conv2d(channel_1, channel_2, kernel_size=1, stride=1, padding=0))
        self.fea_B = Cross_Attn(channel_3)
        self.fea_b = nn.Sequential(nn.ConvTranspose2d(channel_2, channel_2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                   nn.Conv2d(channel_2, channel_3, kernel_size=1, stride=1, padding=0))
        self.fea_C = Cross_Attn(channel_e)
        self.fea_c = nn.Sequential(nn.ConvTranspose2d(channel_3, channel_3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                   nn.Conv2d(channel_3, channel_e, kernel_size=1, stride=1, padding=0))
        self.relu = nn.PReLU()
#        self.SATT = Self_Attn(channel)
        
    def forward(self, a, b, c, e):    
        A = self.fea_A(self.fea_a(a), b) + b
        B = self.fea_B(self.fea_b(A), c) + c
        C = self.fea_C(self.fea_c(B), e) + e
        fuse = self.relu(C)
#        out  = self.SATT(fuse)
        return fuse 
    
    
class AttFuse_8(nn.Module):
    def __init__(self, channel, channel_1, channel_2, channel_3, channel_4, channel_e): 
        super(AttFuse_8, self).__init__()   
        
        self.fea_A = Cross_Attn(channel_2)
        self.fea_a = nn.Sequential(nn.ConvTranspose2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1, output_padding=1),
                                   nn.Conv2d(channel_1, channel_2, kernel_size=1, stride=1, padding=0))
        self.fea_B = Cross_Attn(channel_3)
        self.fea_b = nn.Sequential(nn.ConvTranspose2d(channel_2, channel_2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                   nn.Conv2d(channel_2, channel_3, kernel_size=1, stride=1, padding=0))
        self.fea_C = Cross_Attn(channel_4)
        self.fea_c = nn.Sequential(nn.ConvTranspose2d(channel_3, channel_3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                   nn.Conv2d(channel_3, channel_4, kernel_size=1, stride=1, padding=0))
        self.fea_D = Cross_Attn(channel_e)
        self.fea_d = nn.Sequential(nn.ConvTranspose2d(channel_4, channel_4, kernel_size=3, stride=2, padding=1, output_padding=1),
                                   nn.Conv2d(channel_4, channel_e, kernel_size=1, stride=1, padding=0))
        self.relu = nn.PReLU()
#        self.SATT = Self_Attn(channel)
        
    def forward(self, a, b, c, d, e):    
        A = self.fea_A(self.fea_a(a), b) + b
        B = self.fea_B(self.fea_b(A), c) + c
        C = self.fea_C(self.fea_c(B), d) + d
        D = self.fea_D(self.fea_d(C), e) + e
        fuse = self.relu(D)
#        out  = self.SATT(fuse)
        return fuse 
   

'''''''''生成器'''''''''    
class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(Generator, self).__init__()
        
        '''颜色编码器'''
        self.Pre_Color = Pre_block(in_ch, 16) 
        self.Ecoder_RDB_1 = RD_block(channel=16)  # 256x256
        self.Ecoder_AFM_1 = AttFuse_1(channel=32,  channel_1=16)
        
        self.Ecoder_RDB_2 = RD_block(channel=32)  # 128x128
        self.Ecoder_AFM_2 = AttFuse_2(channel=64,  channel_1=16, channel_2=32)
        
        self.Ecoder_RDB_3 = RD_block(channel=64)  # 64x64
        self.Ecoder_AFM_3 = AttFuse_3(channel=128, channel_1=16, channel_2=32, channel_3=64)
        
        self.Ecoder_RDB_4 = RD_block(channel=128) # 32x32
        self.Ecoder_AFM_4 = AttFuse_4(channel=256, channel_1=16, channel_2=32, channel_3=64, channel_4=128)
        
        self.Ecoder_RDG = RD_group(channel=256)   # 16x16
        
        '''颜色解码器'''
        self.Decoder_AFM_5 = AttFuse_5(channel=128, channel_1=256, channel_e=128)
        self.Decoder_RDB_5 = RD_block(channel=128)
        
        self.Decoder_AFM_6 = AttFuse_6(channel=64, channel_1=256, channel_2=128, channel_e=64)
        self.Decoder_RDB_6 = RD_block(channel=64)
        
        self.Decoder_AFM_7 = AttFuse_7(channel=32, channel_1=256, channel_2=128, channel_3=64, channel_e=32)
        self.Decoder_RDB_7 = RD_block(channel=32)
        
        self.Decoder_AFM_8 = AttFuse_8(channel=16, channel_1=256, channel_2=128, channel_3=64, channel_4=32, channel_e=16)
        self.Decoder_RDB_8 = RD_block(channel=16)
        self.Post_Color = Post_block(16, 3)
        
        '''细节编码器'''
#        self.Pre_Detail = Pre_block(in_ch, 16) 
#        self.Detail_RIB_1 = RI_block(channel=16)  # 256x256
#        self.Detail_AFM_1 = AttFuse_1(channel=32,  channel_1=16)
#        
#        self.Detail_RIB_2 = RI_block(channel=32)  # 128x128
#        self.Detail_AFM_2 = AttFuse_2(channel=64,  channel_1=16, channel_2=32)
#        
#        self.Detail_RIB_3 = RI_block(channel=64)  # 64x64
#        self.Detail_AFM_3 = AttFuse_3(channel=128, channel_1=16, channel_2=32, channel_3=64)
#        
#        self.Detail_RIB_4 = RI_block(channel=128) # 32x32
#        self.Detail_AFM_4 = AttFuse_4(channel=256, channel_1=16, channel_2=32, channel_3=64, channel_4=128)
        
#        self.Detail_RIB = RI_block(channel=256)   # 16x16
        
#        '''细节解码器'''
#        self.Detail_AFM_5 = AttFuse_5(channel=128, channel_1=256, channel_e=128)
#        self.Detail_RIB_5 = RI_block(channel=128)
#        
#        self.Detail_AFM_6 = AttFuse_6(channel=64, channel_1=256, channel_2=128, channel_e=64)
#        self.Detail_RIB_6 = RI_block(channel=64)
#        
#        self.Detail_AFM_7 = AttFuse_7(channel=32, channel_1=256, channel_2=128, channel_3=64, channel_e=32)
#        self.Detail_RIB_7 = RI_block(channel=32)
#        
#        self.Detail_AFM_8 = AttFuse_8(channel=16, channel_1=256, channel_2=128, channel_3=64, channel_4=32, channel_e=16)
#        self.Detail_RIB_8 = RI_block(channel=16)
#        self.Post_Detail = Post_block(16, 3)
        
    def forward(self, haze):
        '''颜色分支'''
        X  = self.Pre_Color(haze)
        X1 = self.Ecoder_RDB_1(X)
        A1 = self.Ecoder_AFM_1(X1)
        
        X2 = self.Ecoder_RDB_2(A1)
        A2 = self.Ecoder_AFM_2(X1,X2)
        
        X3 = self.Ecoder_RDB_3(A2)
        A3 = self.Ecoder_AFM_3(X1,X2,X3)
        
        X4 = self.Ecoder_RDB_4(A3)
        A4 = self.Ecoder_AFM_4(X1,X2,X3,X4)
        
        XX = self.Ecoder_RDG(A4)
        
        A5 = self.Decoder_AFM_5(XX,X4)
        X5 = self.Decoder_RDB_5(A5)

        A6 = self.Decoder_AFM_6(XX,X5,X3)
        X6 = self.Decoder_RDB_6(A6)
        
        A7 = self.Decoder_AFM_7(XX,X5,X6,X2)
        X7 = self.Decoder_RDB_7(A7)
        
        A8 = self.Decoder_AFM_8(XX,X5,X6,X7,X1)
        X8 = self.Decoder_RDB_8(A8) 
        color = self.Post_Color(X8)
        
        '''细节分支'''
#        D  = self.Pre_Detail(haze)
#        D1 = self.Detail_RIB_1(D)
#        B1 = self.Detail_AFM_1(D1)
#        
#        D2 = self.Detail_RIB_2(B1)
#        B2 = self.Detail_AFM_2(D1,D2)
#        
#        D3 = self.Detail_RIB_3(B2)
#        B3 = self.Detail_AFM_3(D1,D2,D3)
#        
#        D4 = self.Detail_RIB_4(B3)
#        B4 = self.Detail_AFM_4(D1,D2,D3,D4)
#        
#        DD = self.Detail_RIB(B4)
#        
#        B5 = self.Detail_AFM_5(DD,D4)
#        D5 = self.Detail_RIB_5(B5)
#
#        B6 = self.Detail_AFM_6(DD,D5,D3)
#        D6 = self.Detail_RIB_6(B6)
#        
#        B7 = self.Detail_AFM_7(DD,D5,D6,D2)
#        D7 = self.Detail_RIB_7(B7)
#        
#        B8 = self.Detail_AFM_8(DD,D5,D6,D7,D1)
#        D8 = self.Detail_RIB_8(B8) 
#        detail = self.Post_Detail(D8)

#        dehaze = color + detail
#        return detail, dehaze
        return color
       