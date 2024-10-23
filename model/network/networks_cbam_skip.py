import torch
import torch.nn.functional as F
from torch import nn
import itertools
from collections import OrderedDict
from . import Interp
from torchvision.models import vgg16

from .attention_module import ChannelAttention, SpatialAttention, CBAMBlock
from .dysample import DySample
from .DANet import DANet, CAM_Module, PAM_Module


class I2INet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, n_downsample=3, skip=True, dim=32, n_res=8, 
                 norm='in', activ='relu', pad_type='reflect', denormalize=False, final_activ='tanh', 
                 attention=False, is_dys=False, cbam_skip=False):
        super(I2INet, self).__init__()

        self.skip=skip
        self.denormalize = denormalize
        self.attention = attention
        self.is_dys = is_dys
        self.cbam_skip = cbam_skip

        # project to feature space
        # 特征映射
        self.conv_in = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)

        # downsampling blocks
        # 下采样块
        self.skip_channels = []
        self.down_blocks = nn.ModuleList()
        for i in range(n_downsample):
            self.down_blocks.append( Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type) )
            dim *= 2
            self.skip_channels.append(dim)

        # residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(n_res):
            self.res_blocks.append( ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type) )

        # attention block
        if self.attention:
            self.attention_block = AttentionBlock(dim, ratio=8, kernel_size=3)
        # if self.danet:
        #     self.danet_block = DANetBlock(256, 256)
        if self.cbam_skip:
            self.cbam_decoder = DecoderCup(
                [512, 256, 128, 64],
                [256, 128, 64, 0],
                [256, 128, 64, 32],
            )
        
        # upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(n_downsample):
            # PyTorch中有两种上采样/下采样的方法，一种是Upsample，另一种是interpolate
            # nn.Upsample 中 scale_factor=2 表示上采样倍数为2，即将特征图的尺寸翻倍
            if self.is_dys:
                self.up_blocks.append( nn.Sequential(
                            DySample(dim, scale=2, style='lp', groups=4, dyscope=False),
                            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)) )
            else:
                self.up_blocks.append( nn.Sequential(
                            nn.Upsample(scale_factor=2),
                            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)) )
            dim //= 2

        # project to image space
        self.conv_out = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=final_activ, pad_type=pad_type)

        #self.apply(weights_init('kaiming'))
        #self.apply(weights_init('gaussian'))


    def forward(self, x):
        # normalize image and save mean/var if using denormalization
        if self.denormalize:
            x_mean = x.view(x.size(0), x.size(1), -1).mean(2).view(x.size(0), x.size(1), 1, 1)
            x_var = x.view(x.size(0), x.size(1), -1).var(2).view(x.size(0), x.size(1), 1, 1)
            x = (x-x_mean)/x_var

        # project to feature space
        x = self.conv_in(x)

        # downsampling blocks
        xs = []
        for block in self.down_blocks:
            x = block(x)
            xs += [x]

        # residual blocks
        x_att = []
        x_att += [x]
        for block in self.res_blocks:
            x = block(x)
        x_att += [x]

        # attention module
        # if self.danet:
        #     x = self.danet_block(x_att)
        # elif self.attention:
        #     x = self.attention_block(x_att)
        if self.attention:
            x = self.attention_block(x_att)

        # upsampling blocks
        if self.cbam_skip:
            x = self.cbam_decoder(x, xs[-2::-1])
        else:
            for block, skip in zip(self.up_blocks, reversed(xs)):
                x = block(x)
                if self.skip:
                    x = x + skip
                

        # project to image space
        x = self.conv_out(x)

        # denormalize if necessary
        if self.denormalize:
            x = x*x_var+x_mean
        return x


class CCNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, layers=5, dim=32, norm='gn', activ='relu', pad_type='reflect', final_activ='tanh'):
        super(CCNet, self).__init__()
        self.model = []
        #self.model += [Conv2dBlock(input_dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(input_dim, dim, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(layers-2):
            self.model += [Conv2dBlock(dim, dim, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, output_dim, 1, 1, 0, norm='none', activation=final_activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
    def forward(self, x):
        return self.model(x)


class vgg_features(nn.Module):
    def __init__(self):
        super(vgg_features, self).__init__()
        # get vgg16 features up to conv 4_3
        self.model = nn.Sequential(*list(vgg16(pretrained=True).features)[:23])
        # will not need to compute gradients
        for param in self.parameters():
            param.requires_grad=False

    def forward(self, x, renormalize=True):
        # change normaliztion form [-1,1] to VGG normalization
        if renormalize:
            x = ((x*.5+.5)-torch.cuda.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))/torch.cuda.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1)
            # x = ((x*.5+.5)-torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))/torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1)

        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, gan_type='lsgan', input_dim=3, dim=64, n_layers=4, norm='bn', activ='lrelu', pad_type='reflect'):
        super(Discriminator, self).__init__()
        self.gan_type = gan_type
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 4, 2, 1, norm='none', activation=activ, pad_type=pad_type)]
        for i in range(n_layers - 1):
            self.model += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [nn.Conv2d(dim, 1, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        #self.apply(weights_init('gaussian'))

    def forward(self, input):
        return self.model(input).mean(3).mean(2).squeeze()

    def calc_dis_loss(self, input_fake, input_real):
        input_fake = input_fake.detach()
        input_real = input_real.detach()
        out0 = self.forward(input_fake)
        out1 = self.forward(input_real)
        if self.gan_type == 'lsgan':
            loss = torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
        elif self.gan_type == 'nsgan':
            all0 = torch.zeros_like(out0, requires_grad=False).cuda()
            all1 = torch.ones_like(out1, requires_grad=False).cuda()
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                              F.binary_cross_entropy(F.sigmoid(out1), all1))
        elif self.gan_type == 'wgan':
            loss = out0.mean()-out1.mean()
            # grad penalty
            BatchSize = input_fake.size(0)
            alpha = torch.rand(BatchSize,1,1,1, requires_grad=False).cuda()
            interpolates = (alpha * input_real) + (( 1 - alpha ) * input_fake)
            interpolates.requires_grad=True
            outi = self.forward(interpolates)
            all1 = torch.ones_like(out1, requires_grad=False).cuda()
            gradients = torch.autograd.grad(outi, interpolates, grad_outputs=all1, create_graph=True)[0]
            #gradient_penalty = ((gradients.view(BatchSize,-1).norm(2, dim=1) - 1) ** 2).mean()
            gradient_penalty = ((gradients.view(BatchSize,-1).norm(1, dim=1) - 1).clamp(0) ** 2).mean()
            loss += 10*gradient_penalty
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        out0 = self.forward(input_fake)
        if self.gan_type == 'lsgan':
            loss = torch.mean((out0 - 1)**2)
        elif self.gan_type == 'nsgan':
            all1 = torch.ones_like(out0.data, requires_grad=False).cuda()
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
        elif self.gan_type == 'wgan':
            loss = -out0.mean()
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


class DecoderCup(nn.Module):
    def __init__(self, 
                 in_channels, 
                 skip_channels, 
                 out_channels):
        super().__init__()

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, features=None):
        
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < len(features)) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.cbam1 = CBAMBlock(32, ratio=4, kernel_size=3)
        self.cbam2 = CBAMBlock(64, ratio=4, kernel_size=3)
        self.cbam3 = CBAMBlock(128, ratio=4, kernel_size=3)
        
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if skip.size(1) and x.size(1) == 64:
                skip = self.cbam1(skip) 
            
            if skip.size(1) and x.size(1) == 128:
                skip = self.cbam2(skip)
                
            if skip.size(1) and x.size(1) == 256:
                skip = self.cbam3(skip)
                
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x



##################################################################################
# 解码块
##################################################################################
class DASkipBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.da = DANet(skip_channels, skip_channels, reduction=8)
            
        
    def forward(self, x, skip=None):
        if skip is not None:
            skip = self.da(skip)
                
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

##################################################################################
# Attention Blocks
##################################################################################
class AttentionBlock(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(AttentionBlock, self).__init__()
        self.channelattention = ChannelAttention(channel*2, ratio=ratio)
        self.spatialattention1 = SpatialAttention(kernel_size=kernel_size)
        self.spatialattention2 = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x_att):
        x = torch.cat(x_att, 1)
        x = x * self.channelattention(x)
        
        # 分别使用 空间注意力机制
        x1 = x[:, :x_att[0].size(1), :, :]
        x2 = x[:, x_att[0].size(1):, :, :]
        x1 = x1 * self.spatialattention1(x1)
        x2 = x2 * self.spatialattention2(x2)
        
        return x1 + x2
    
##################################################################################
# DA_Net Block
##################################################################################
class DANetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(DANetBlock, self).__init__()
        inter_channels = in_channels // reduction
        
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels*2, inter_channels*2, 3, padding=1, bias=False),
                                    norm(inter_channels*2),
                                    nn.ReLU())
        self.sa = PAM_Module(inter_channels*2)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels*2, inter_channels*2, 3, padding=1, bias=False),
                                    norm(inter_channels*2),
                                    nn.ReLU())
        self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), 
                                   nn.Conv2d(inter_channels*2, out_channels*2, 1),
                                   nn.ReLU())
        
        self.conv5c1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        self.sc1 = CAM_Module(inter_channels)
        self.conv521 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        
        self.conv5c2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        self.sc2 = CAM_Module(inter_channels)
        self.conv522 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())


        self.conv71 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())
        self.conv72 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), 
                                   nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())

    def forward(self, x_att):
        x = torch.cat(x_att, 1)
        
        # PAM
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)
        
        # 分别使用 空间注意力机制
        # part1, part2 = torch.split(feature_map, split_size_or_sections=1, dim=1)
        x1 = sa_output[:, :x_att[0].size(1), :, :]
        x2 = sa_output[:, x_att[0].size(1):, :, :]
        
        # CAM
        feat21 = self.conv5c1(x1)
        sc_feat1 = self.sc1(feat21)
        sc_conv1 = self.conv521(sc_feat1)
        sc_output1 = self.conv71(sc_conv1)
        
        feat22 = self.conv5c2(x2)
        sc_feat2 = self.sc2(feat22)
        sc_conv2 = self.conv522(sc_feat2)
        sc_output2 = self.conv72(sc_conv2)
        
        feat_sum = sc_conv1 + sc_conv2

        sasc_output = self.conv8(feat_sum)
        
        return sasc_output

def norm(planes, mode='bn', groups=16):
    if mode == 'bn':
        return nn.BatchNorm2d(planes, momentum=0.95, eps=1e-03)
    elif mode == 'gn':
        return nn.GroupNorm(groups, planes)
    else:
        return nn.Sequential()


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', transposed=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'gn':
            # 将channel切分成许多组进行归一化，
            # 将dim分成8组，每组的channel数相同
            # torch.nn.GroupNorm(num_groups,num_channels)
            self.norm = nn.GroupNorm(norm_dim//8, norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transposed:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x




##################################################################################
# weight initialization
##################################################################################

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                nn.init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant(m.bias.data, 0.0)
    return init_fun


if __name__ == '__main__':
    model = Model()
    print(model)

