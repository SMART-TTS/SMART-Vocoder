import torch
from torch import nn
from wavenet import WaveNetAR, WaveNetGlow2
from math import log, pi, sqrt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal


class SqueezeLayer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x, ):
        x = self.squeeze(x, self.scale)
        x = x.unsqueeze(1)
        return x

    def reverse(self, z):
        z = z.squeeze(1)
        z = self.unsqueeze(z, self.scale)

        return z

    def squeeze(self, x, scale):
        B, C, T = x.size()
        squeezed_x = x.contiguous().view(B, C, T // scale, scale).permute(0, 1, 3, 2)
        squeezed_x = squeezed_x.contiguous().view(B, C * scale, T // scale)

        return squeezed_x

    def unsqueeze(self, z, scale):
        B, C, T = z.size()
        unsqueezed_z = z.view(B, C // scale, scale, T).permute(0, 1, 3, 2)
        unsqueezed_z = unsqueezed_z.contiguous().view(B, C // scale, T * scale)

        return unsqueezed_z


class Invertible1x1Conv(torch.nn.Module):
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, c, log_det):
        # shape
        B_, C_, T_ = z.size()
        W = self.conv.weight.squeeze()
        log_det_W = B_ * T_ * torch.logdet(W)
        log_det_W = log_det_W.sum()
        z = self.conv(z)
        log_det = log_det + log_det_W

        return z, c, log_det

    def reverse(self, z, c):
        W = self.conv.weight.squeeze()
        W_inverse = W.float().inverse()
        W_inverse = Variable(W_inverse[..., None])
        self.W_inverse = W_inverse
        z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
        
        return z, c


class ActNorm(nn.Module):
    def __init__(self, in_channels, pretrained):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1))

        self.initialized = pretrained

    def initialize(self, x):
        flatten = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        mean = (
            flatten.mean(1)
            .unsqueeze(1)
            .unsqueeze(2)
            .permute(1, 0, 2)
        )
        std = (
            flatten.std(1)
            .unsqueeze(1)
            .unsqueeze(2)
            .permute(1, 0, 2)
        )

        self.loc.data.copy_(-mean)
        self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x, c, log_det):
        if not self.initialized:
            self.initialize(x)
            self.initialized = True

        z = self.scale * (x + self.loc)

        log_abs = torch.log(torch.abs(self.scale))
        _B, _, _T = x.size()
        log_det += log_abs.sum() * _B * _T

        return z, c, log_det

    def reverse(self, z, c):
        x = (z / self.scale) - self.loc

        return x, c


class PosConditionedFlow(nn.Module):
    def __init__(self, in_channels, cin_channels, dilation, pos_group, n_channels, n_layers):
        super().__init__()
        self.pos_group = pos_group
        self.in_channels = in_channels
        self.WN = WaveNetGlow2(in_channels//2, cin_channels, dilation, pos_group, n_channels, n_layers)

    def forward(self, x, c, log_det):
        if self.pos_group > 1:
            B_orig = x.shape[0] // self.pos_group
            pos = torch.tensor(range(self.pos_group)).to(x.device).repeat(B_orig)
        else:
            pos = None

        x_a, x_b = x.chunk(2,1)
        log_s, b = self.WN(x_a, c, pos)
        x_b = torch.exp(log_s) * x_b + b
        log_det = log_det + log_s.sum()
        z = torch.cat((x_a, x_b), dim=1)

        return z, c, log_det

    def reverse(self, z, c):
        if self.pos_group > 1:
            B_orig = z.shape[0] // self.pos_group
            pos = torch.tensor(range(self.pos_group)).to(z.device).repeat(B_orig)
        else:
            pos = None

        z_a, z_b = z.chunk(2,1)
        log_s, b = self.WN(z_a, c, pos)
        z_b = torch.exp(-log_s) * (z_b - b)

        x = torch.cat((z_a, z_b), dim=1)

        return x, c


class EqualResolutionBlock(nn.Module):
    def __init__(self, chains):
        super().__init__()
        self.chains = nn.ModuleList(chains)

    def forward(self, x, c, log_det):
        for chain in self.chains:
            x, c, log_det = chain(x, c, log_det)
        z = x

        return z, c, log_det

    def reverse(self, z, c):
        for chain in self.chains[::-1]:
            z, c = chain.reverse(z, c)
        x = z

        return x, c


class SmartVocoder(nn.Module):
    def __init__(self, hps):
        super().__init__()

        in_channels = 1  # number of channels in audio
        cin_channels = 80  # number of channels in mel-spectrogram (freq. axis)

        self.n_ER_blocks = hps.n_ER_blocks
        self.n_flow_blocks = hps.n_flow_blocks
        self.n_layers = hps.n_layers
        self.n_channels = hps.n_channels

        self.pretrained = hps.pretrained
        
        self.sqz_layer = SqueezeLayer(hps.sqz_scale_i)
        self.ER_blocks = nn.ModuleList()

        self.upsample_conv = nn.ModuleList()
        for s in [4, 4, 4]:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))
        
        in_channels *= hps.sqz_scale_i
        for i in range(self.n_ER_blocks):
            dilation_base = 2 if i < 2 else 1
            pos_group = hps.sqz_scale ** (i-1)
            self.ER_blocks.append(self.build_ER_block(hps.n_flow_blocks, in_channels, cin_channels, dilation_base, 
                                pos_group, hps.n_channels, hps.n_layers, hps.pretrained))
            in_channels *= hps.sqz_scale
        
    def build_ER_block(self, n_flow_blocks, in_channels, cin_channels, di_base, pos_group, n_channels, n_layers, pretrained):
        chains = []
        for _ in range(n_flow_blocks):
            chains += [ActNorm(in_channels, pretrained=pretrained)]
            chains += [Invertible1x1Conv(in_channels)]
            chains += [PosConditionedFlow(in_channels, cin_channels, di_base, pos_group, n_channels, n_layers)]

        ER_block = EqualResolutionBlock(chains)

        return ER_block

    def forward(self, x, mel):
        Bx, Cx, Tx = x.size()
        c = mel.unsqueeze(1)
        c_list = [mel]
        for i, conv in enumerate(self.upsample_conv):
            c = conv(c)
            if i % 2 == 1:
                c_list.append(c.squeeze(1))
        c_list = c_list[::-1]
        
        out = self.sqz_layer(x).squeeze(1)
        log_det = 0
        c_in = c_list[0]
        for i, block in enumerate(self.Glow_blocks):
            out, _, log_det = block(out, c_in, log_det)

            if i != len(self.Glow_blocks) -1 :
                B, C, T = out.shape
                out = out.view(B, C, T//4, 4).permute(0,1,3,2).contiguous().view(4*B,C,T//4)
                
                Bc, Cc, Tc = c_list[i+1].shape
                c_in = c_list[i+1].repeat(1, 4 ** (i+1), 1).view(-1, Cc, Tc)
        z = out
        
        log_p_sum = 0.5 * (- log(2.0 * pi) - z.pow(2)).sum()
        log_det = log_det.sum() / (Bx * Cx * Tx)
        log_p = log_p_sum / (Bx * Cx * Tx)

        return log_p, log_det

    def reverse(self, z, mel):
        c = mel.unsqueeze(1)
        c_list = [mel]
        for i, conv in enumerate(self.upsample_conv):
            c = conv(c)
            if i % 2 == 1:
                c_list.append(c.squeeze(1))

        out = self.sqz_layer(z).squeeze(1)
        for i in range(len(self.Glow_blocks)-1):
            B, C, T = out.shape
            out = out.view(B, C, T//4, 4).permute(0,1,3,2).contiguous().view(4*B,C,T//4)

        Bc, Cc, Tc = c_list[0].shape
        c_in = c_list[0].repeat(1, 4 ** (len(self.Glow_blocks)-1), 1).view(-1, Cc, Tc)
        for i, block in enumerate(self.Glow_blocks[::-1]):
            out, _ = block.reverse(out, c_in)

            if i != len(self.Glow_blocks)-1 :
                B, C, T = out.shape
                out = out.view(B//4, C, 4, T).permute(0,1,3,2).contiguous().view(B//4, C, 4 * T)
                Bc, Cc, Tc = c_list[i+1].shape
                c_in = c_list[i+1].repeat(1, 4 ** (len(self.Glow_blocks)-2-i), 1).view(-1, Cc, Tc)
            
        out = out.unsqueeze(1)
        x = self.sqz_layer.reverse(out)

        return x

    def upsample(self, c):
        for f in self.upsample_conv:
            c = f(c)
        return c

