import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg16

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

    def calc_gen_losses(self, input_fake):
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

