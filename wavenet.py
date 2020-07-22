import torch
import torch.nn as nn
import math
from math import log2
import torch.nn.functional as F


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_h=1, dilation_w=1, AR_dir=0, t_cond=False):
        super(Conv2D, self).__init__()
        if AR_dir == 0 and not t_cond:
            padding_h = int((kernel_size*dilation_h - dilation_h))
            padding_w =  int((kernel_size*dilation_w - dilation_w)/2) 

        if AR_dir == 1 and not t_cond:
            padding_h = int((kernel_size*dilation_h - dilation_h)/2)
            padding_w =  int((kernel_size*dilation_w - dilation_w)) 

        if t_cond:
            padding_h = int((kernel_size*dilation_h - dilation_h)/2)
            padding_w =  int((kernel_size*dilation_w - dilation_w)/2) 
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              dilation=(dilation_h, dilation_w), padding=(self.padding_h, self.padding_w))
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        B, C, H, W = tensor.shape
        out = self.conv(tensor)
        out = out[:,:,:H,:W]

        return out


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, x):
        out = self.conv(x)
        out = out * torch.exp(self.scale * 3)
        return out


class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size,
                 cin_channels=None, AR_dir=0, t_cond=True, dilation_h=None, dilation_w=None):
        super(ResBlock2D, self).__init__()
        self.t_cond = t_cond
        self.cin_channels = cin_channels
        self.skip = True if skip_channels is not None else False

        self.filter_conv = Conv2D(in_channels, out_channels, kernel_size, dilation_h, dilation_w, AR_dir, t_cond)
        self.gate_conv = Conv2D(in_channels, out_channels, kernel_size, dilation_h, dilation_w, AR_dir, t_cond)
        self.res_conv = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        self.filter_conv_c = nn.Conv2d(cin_channels, out_channels, kernel_size=1)
        self.gate_conv_c = nn.Conv2d(cin_channels, out_channels, kernel_size=1)
        self.filter_conv_c = nn.utils.weight_norm(self.filter_conv_c)
        self.gate_conv_c = nn.utils.weight_norm(self.gate_conv_c)
        nn.init.kaiming_normal_(self.filter_conv_c.weight)
        nn.init.kaiming_normal_(self.gate_conv_c.weight)

        if self.skip:
            self.skip_conv = nn.Conv2d(out_channels, skip_channels, kernel_size=1)
            self.skip_conv = nn.utils.weight_norm(self.skip_conv)
            nn.init.kaiming_normal_(self.skip_conv.weight)


        if self.t_cond:
            self.filter_linear_t = nn.Linear(1, out_channels)
            self.gate_linear_t = nn.Linear(1, out_channels)
            self.filter_linear_t = nn.utils.weight_norm(self.filter_linear_t)
            self.gate_linear_t = nn.utils.weight_norm(self.gate_linear_t)
            nn.init.kaiming_normal_(self.filter_linear_t.weight)
            nn.init.kaiming_normal_(self.gate_linear_t.weight)

    def forward(self, tensor, c, t=None):

        h_filter = self.filter_conv(tensor) + self.filter_conv_c(c)
        h_gate = self.gate_conv(tensor) + self.gate_conv_c(c)

        if t is not None:
            B, C, H, W = tensor.shape
            h_filter = h_filter + self.filter_linear_t(t).view(B,C,1,1)
            h_gate = h_gate + self.gate_linear_t(t).view(B,C,1,1)

        out = torch.tanh(h_filter) * torch.sigmoid(h_gate)

        res = self.res_conv(out)
        skip = self.skip_conv(out) if self.skip else None
        return (tensor + res) * math.sqrt(0.5), skip


class WaveNet(nn.Module):
    # a variant of WaveNet-like arch that operates on 2D feature for WF
    def __init__(self, in_channels, cin_channels, di_base_h, di_base_w, wvn_channel, wvn_layer, AR_dir=0, t_cond=False,kernel_size=3):
        super().__init__()

        residual_channels = wvn_channel
        gate_channels = wvn_channel
        skip_channels = wvn_channel
        filter_size = wvn_channel
        
        self.t_cond = t_cond
        self.skip = True 

        self.dilation_h = []
        self.dilation_w = []
        
        for i in range(wvn_layer):
            self.dilation_h.append(di_base_h ** i)
            self.dilation_w.append(di_base_w ** i)

        self.front_conv = nn.Sequential(
            Conv2D(in_channels, residual_channels, 1, 1, 1),
        )

        self.res_blocks = nn.ModuleList()
        for n in range(wvn_layer):
            self.res_blocks.append(ResBlock2D(residual_channels, gate_channels, skip_channels, kernel_size,
                                              cin_channels=cin_channels, AR_dir=AR_dir, t_cond=t_cond,
                                              dilation_h=self.dilation_h[n], dilation_w=self.dilation_w[n]))

        self.final_conv = nn.Sequential(
            nn.ReLU(),
            Conv2D(residual_channels, filter_size, 1, 1, 1),
            nn.ReLU(),
        )

        if self.t_cond:
            self.proj_end = ZeroConv2d(filter_size, in_channels)
        else:
            self.proj_log_s = ZeroConv2d(filter_size, in_channels)
            self.proj_b = ZeroConv2d(filter_size, in_channels)

    def forward(self, x, c, t=None):
        h = self.front_conv(x)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            if self.skip:
                h, s = f(h, c, t)
                skip += s
            else:
                h, _ = f(h, c, t)
        if self.skip:
            out = self.final_conv(skip)
        else:
            out = self.final_conv(h)

        if self.t_cond:
            dxdt = self.proj_end(out)
            return dxdt

        else:
            log_s = self.proj_log_s(out)
            b = self.proj_b(out)
            return log_s, b


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels, audio_shape=None):
    if audio_shape is not None:
        B, C, H, W = audio_shape
        input_a = input_a[:,:,:H,:]
    n_channels_int = n_channels[0]
    in_act = input_a + input_b 
    t_act = torch.tanh(in_act[:, :n_channels_int])
    s_act = torch.sigmoid(in_act[:, n_channels_int:])
    acts = t_act * s_act
    return acts

def fused_add_tanh_sigmoid_multiply_AR(input_a, input_b, input_c, n_channels, audio_shape=None):
    if audio_shape is not None:
        B, C, H, W = audio_shape
        input_a = input_a[:,:,:H,:]
    n_channels_int = n_channels[0]

    in_act = input_a + input_b +input_c
    t_act = torch.tanh(in_act[:, :n_channels_int])
    s_act = torch.sigmoid(in_act[:, n_channels_int:])
    acts = t_act * s_act
    return acts


def fused_add_tanh_sigmoid_multiply2(input_a, input_b, input_c, n_channels, audio_shape=None):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b + input_c
    t_act = torch.tanh(in_act[:, :n_channels_int])
    s_act = torch.sigmoid(in_act[:, n_channels_int:])
    acts = t_act * s_act
    return acts


def fused_add_tanh_sigmoid_multiply3(n_channels, input_a, input_b, input_c=None):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    if input_c is not None:
        in_act = in_act + input_c
    t_act = torch.tanh(in_act[:, :n_channels_int])
    s_act = torch.sigmoid(in_act[:, n_channels_int:])
    acts = t_act * s_act
    return acts

def fused_add_tanh_sigmoid_multiply6(n_channels, aud_list, input_b, input_c=None):
    n_channels_int = n_channels[0]
    t_act_list = []
    s_act_list = []
    for input_a in aud_list:
        in_act = input_a + input_b
        if input_c is not None:
            in_act = in_act + input_c
    
        t_act_list.append(in_act[:, :n_channels_int])
        s_act_list.append(in_act[:, n_channels_int:])

    t_act = 0.0
    s_act = 0.0

    for i in range(len(t_act_list)):
        t_act = t_act + t_act_list[i]
        s_act = s_act + s_act_list[i]

    t_act = torch.tanh(t_act)
    s_act = torch.sigmoid(s_act)

    acts = t_act * s_act

    return acts

def fused_add_tanh_sigmoid_multiply4(input_a, input_b, input_c, n_channels):
    n_channels_int = n_channels[0]
    if input_c is None:
        in_act = input_a + input_b
    else:
        in_act = input_a + input_b + input_c
    t_act = torch.tanh(in_act[:, :n_channels_int])
    s_act = torch.sigmoid(in_act[:, n_channels_int:])
    acts = t_act * s_act
    return acts
def fused_add_tanh_sigmoid_multiply5(input_a, input_b, input_c, input_d, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b

    if input_c is not None:
        in_act = in_act + input_c
    
    if input_d is not None:
        in_act = in_act + input_d

    t_act = torch.tanh(in_act[:, :n_channels_int])
    s_act = torch.sigmoid(in_act[:, n_channels_int:])
    acts = t_act * s_act
    return acts

class WaveNetAR(nn.Module):
    def __init__(self, in_channels, cin_channels, scale_h, di_base_h, di_base_w, n_channels, n_layers, AR_dir=0, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        assert(AR_dir == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv2d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv2d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        self.pos_emb = torch.nn.Embedding(scale_h-1, 32)
        pos_layer = torch.nn.Linear(32, 2*n_channels*n_layers)
        self.pos_layer = torch.nn.utils.weight_norm(pos_layer, name='weight')

        for i in range(n_layers):
            dilation_h = di_base_h ** i
            dilation_w = di_base_w ** i

            padding_h = int((kernel_size*dilation_h - dilation_h))
            padding_w =  int((kernel_size*dilation_w - dilation_w)/2) 

            in_layer = torch.nn.Conv2d(n_channels, 2*n_channels, kernel_size,
                                       dilation=[dilation_h, dilation_w], padding=[padding_h, padding_w])
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv2d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, pos):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect).unsqueeze(2)
        pos = self.pos_emb(pos)
        pos = self.pos_layer(pos)
        pos = pos.permute(1,0).contiguous()
        Cp, Hp = pos.shape
        pos = pos.view(1,Cp,Hp, 1)
        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            acts = fused_add_tanh_sigmoid_multiply_AR(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset+2*self.n_channels, :], 
                pos[:,spect_offset:spect_offset+2*self.n_channels],
                n_channels_tensor,
                audio_shape)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)


class WaveNetGlow(nn.Module):
    def __init__(self, in_channels, cin_channels, di_base, pos_group, n_channels, n_layers, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        pos_emb = torch.nn.Embedding(pos_group, 128)
        pos_layer = torch.nn.Linear(128, 2*n_channels*n_layers)
        self.pos_layer = torch.nn.utils.weight_norm(pos_layer, name='weight')
        self.pos_emb = torch.nn.utils.weight_norm(pos_emb, name='weight')

        for i in range(n_layers):
            dilation = di_base ** i

            padding = int((kernel_size*dilation - dilation)) // 2

            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, pos):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        Bsp, Csp, Tsp = spect.shape
        spect = spect.repeat(1,self.pos_group,1).view(-1,Csp,Tsp)

        pos = F.relu(self.pos_emb(pos))
        pos = self.pos_layer(pos)
        pos = pos.unsqueeze(2)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            acts = fused_add_tanh_sigmoid_multiply2(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset+2*self.n_channels, :], 
                pos[:, spect_offset:spect_offset+2*self.n_channels], 
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)

class WaveNetGlow4(nn.Module):
    def __init__(self, in_channels, cin_channels, prev_cond, di_base, pos_group, n_channels, n_layers, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        if prev_cond:
            prev_cond_layer = torch.nn.Conv3d(1,  2*n_channels*n_layers,  [3,4,1], padding=[3, 0, 0])
            self.prev_cond_layer = torch.nn.utils.weight_norm(prev_cond_layer, name='weight')

        for i in range(n_layers):
            dilation = di_base ** i

            padding = int((kernel_size*dilation - dilation)) // 2

            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, x_prev):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        Bsp, Csp, Tsp = spect.shape
        # spect_orig = spect.clone().detach()
        # print(spect_orig.shape)
        if x_prev is not None:
            B, C1, C2, H, W = x_prev.shape
            x_prev = self.prev_cond_layer(x_prev)
            x_prev = x_prev[:,:,:4,:,:].permute(0,2,1,3,4).contiguous().view(B*C2, -1, W)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            spect_in = spect[:, spect_offset:spect_offset+2*self.n_channels, :]
            if x_prev is not None:
                x_prev_in = x_prev[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                x_prev_in = None

            acts = fused_add_tanh_sigmoid_multiply3(
                n_channels_tensor,
                self.in_layers[i](audio),
                spect_in, 
                x_prev_in)
            

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)

class WaveNetGlow3(nn.Module):
    def __init__(self, in_channels, cin_channels, prev_cond, di_base, pos_group, n_channels, n_layers, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        if prev_cond:
            prev_cond_layer = torch.nn.Conv3d(1, 2*n_channels*n_layers, [3,4,3], padding=[3, 0, 1])
            self.prev_cond_layer = torch.nn.utils.weight_norm(prev_cond_layer, name='weight')

        for i in range(n_layers):
            dilation = di_base ** i

            padding = int((kernel_size*dilation - dilation)) // 2

            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, x_prev):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        Bsp, Csp, Tsp = spect.shape
        # spect_orig = spect.clone().detach()
        # print(spect_orig.shape)
        if x_prev is not None:
            print('audio', audio.shape)
            B, C1, C2, H, W = x_prev.shape
            print('x_prev before', x_prev.shape)
            x_prev = self.prev_cond_layer(x_prev)
            x_prev = x_prev[:,:,:4,:,:].permute(0,2,1,3,4).contiguous().view(B*C2, -1, W)
            print('x_prev after', x_prev.shape)


        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            spect_in = spect[:, spect_offset:spect_offset+2*self.n_channels, :]
            if x_prev is not None:
                x_prev_in = x_prev[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                x_prev_in = None

            acts = fused_add_tanh_sigmoid_multiply3(
                n_channels_tensor,
                self.in_layers[i](audio),
                spect_in, 
                x_prev_in)
            

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)



class WaveNetGlow7(nn.Module):
    def __init__(self, in_channels, cin_channels, prev_cond, di_base, pos_group, n_channels, n_layers, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        if prev_cond:
            self.prev_cond_layer1 = torch.nn.Conv3d(1, n_channels, [3,1,1], padding=[0, 0, 0])
            self.prev_cond_layer2 = torch.nn.Conv3d(n_channels, n_channels, [1,1,1], padding=[0, 0, 0])
            self.prev_cond_layer3 = torch.nn.Conv3d(n_channels, 2*n_channels*n_layers, [1,1,1], padding=[0, 0, 0])

        for i in range(n_layers):
            dilation = di_base ** i

            padding = int((kernel_size*dilation - dilation)) // 2

            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, x_prev=None, reverse=False):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        if x_prev is not None:
            if not reverse:
                # print('spect', spect.shape)
                # print('prev in', x_prev.shape)
                x_prev = F.relu(self.prev_cond_layer1(x_prev))
                x_prev = F.relu(self.prev_cond_layer2(x_prev))
                # print('mid', x_prev.shape)
                x_prev = self.prev_cond_layer3(x_prev).mean(dim=3).squeeze()


            else:
                B, C, T = audio_shape
                prev_conv_weight = self.prev_cond_layer1.weight
                prev_conv_bias = self.prev_cond_layer1.bias
                # print('x_prev befo', x_prev.shape)
                x_prev = F.relu(F.conv3d(x_prev, prev_conv_weight, prev_conv_bias, padding=[3, 0, 0])[:,:, :4])
                x_prev = F.relu(self.prev_cond_layer2(x_prev))
                x_prev = self.prev_cond_layer3(x_prev)

                # print('mid', x_prev.shape)
                x_prev = x_prev.mean(dim=3)
                # print('mid2', x_prev.shape)
                x_prev = x_prev.permute(0,2,1,3).contiguous().view(B, -1, T)
                # print('x_prev', x_prev.shape)
                # x_prev = x_prev.permute(0,2,1,3,4).contiguous().view(B, -1, T)
                # print('audio', audio.shape)
                # print('x_prev', x_prev.shape)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            spect_in = spect[:, spect_offset:spect_offset+2*self.n_channels, :]
            if x_prev is not None:
                x_prev_in = x_prev[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                x_prev_in = None

            acts = fused_add_tanh_sigmoid_multiply3(
                n_channels_tensor,
                self.in_layers[i](audio),
                spect_in, 
                x_prev_in)
            

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)


class WaveNetGlow6(nn.Module):
    def __init__(self, in_channels, cin_channels, prev_cond, di_base, pos_group, n_channels, n_layers, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        if prev_cond:
            self.prev_cond_layer = torch.nn.Conv3d(1, 2*n_channels*n_layers, [3,4,3], padding=[0, 0, 1])

        for i in range(n_layers):
            dilation = di_base ** i

            padding = int((kernel_size*dilation - dilation)) // 2

            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, x_prev=None, reverse=False):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        if x_prev is not None:
            if not reverse:
                x_prev = self.prev_cond_layer(x_prev).squeeze()
                # print(x_prev.shape)
            else:
                B, C, T = audio_shape
                prev_conv_weight = self.prev_cond_layer.weight
                prev_conv_bias = self.prev_cond_layer.bias
                # print('x_prev befo', x_prev.shape)
                x_prev = F.conv3d(x_prev, prev_conv_weight, prev_conv_bias, padding=[3, 0, 1])[:,:, :4]
                x_prev = x_prev.permute(0,2,1,3,4).contiguous().view(B, -1, T)
                # print('audio', audio.shape)
                # print('x_prev', x_prev.shape)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            spect_in = spect[:, spect_offset:spect_offset+2*self.n_channels, :]
            if x_prev is not None:
                x_prev_in = x_prev[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                x_prev_in = None

            acts = fused_add_tanh_sigmoid_multiply3(
                n_channels_tensor,
                self.in_layers[i](audio),
                spect_in, 
                x_prev_in)
            

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)


class WaveNetGlow5(nn.Module):
    def __init__(self, in_channels, cin_channels, di_cycle, pos_group, n_channels, n_layers, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        pos_emb = torch.nn.Embedding(pos_group, 32)
        pos_layer = torch.nn.Linear(32, 2*n_channels*n_layers)
        self.pos_layer = torch.nn.utils.weight_norm(pos_layer, name='weight')
        self.pos_emb = torch.nn.utils.weight_norm(pos_emb, name='weight')

        for i in range(n_layers):
            dilation = 2 ** (i % di_cycle)
            print('dilation', dilation)

            padding = int((kernel_size*dilation - dilation)) // 2

            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, pos):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        Bsp, Csp, Tsp = spect.shape
        # spect_orig = spect.clone().detach()
        # print(spect_orig.shape)

        pos = F.relu(self.pos_emb(pos))
        pos = self.pos_layer(pos)
        pos = pos.unsqueeze(2)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            acts = fused_add_tanh_sigmoid_multiply2(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset+2*self.n_channels, :], 
                pos[:, spect_offset:spect_offset+2*self.n_channels], 
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)


class WaveNetGlow8(nn.Module):
    def __init__(self, in_channels, cin_channels, prev_cond, di_base, pos_group, n_channels, n_layers, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group
        self.prev_cond = prev_cond

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        if prev_cond:
            self.prev_cond_layer1 = torch.nn.Conv3d(1, 2*n_channels*n_layers, [3,4,3], padding=[0, 0, 1])
            # self.prev_cond_layer2 = torch.nn.Conv3d(n_channels, 2*n_channels*n_layers, [1,1,1], padding=[0, 0, 0])

        for i in range(n_layers):
            dilation = di_base ** i

            padding = int((kernel_size*dilation - dilation)) // 2

            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, x_prev=None, reverse=False):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        Bsp, Csp, Tsp = spect.shape
        # spect_orig = spect.clone().detach()
        # print(spect_orig.shape)

        if x_prev is not None:
            if not reverse:
                x_prev = (self.prev_cond_layer1(x_prev))
                # x_prev = self.prev_cond_layer2(x_prev)
                x_prev = x_prev.squeeze()

            else:
                B, C, T = audio_shape
                prev_conv_weight = self.prev_cond_layer1.weight
                prev_conv_bias = self.prev_cond_layer1.bias
                x_prev = F.conv3d(x_prev, prev_conv_weight, prev_conv_bias, padding=[3, 0, 1])[:,:, :4].squeeze(3)
                x_prev = x_prev.permute(0,2,1,3).contiguous().view(B, -1, T)


        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            if x_prev is not None:
                x_prev_in = x_prev[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                x_prev_in = None

            acts = fused_add_tanh_sigmoid_multiply4(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset+2*self.n_channels, :], 
                x_prev_in,
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)




class WaveNetGlow12(nn.Module):
    def __init__(self, in_channels, cin_channels, di_base, pos_group, n_channels, n_layers, prev_cond=False, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        self.pos_group = pos_group
        if pos_group > 1:
            emb_ch = pos_group
            pos_emb = torch.nn.Embedding(pos_group, emb_ch)
            pos_layer = torch.nn.Linear(emb_ch, 2*n_channels*n_layers)
            self.pos_layer = torch.nn.utils.weight_norm(pos_layer, name='weight')
            self.pos_emb = torch.nn.utils.weight_norm(pos_emb, name='weight')
       
        for i in range(n_layers):
            dilation = di_base ** i

            padding = int((kernel_size*dilation - dilation)) // 2

            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, pos=None, x_prev=None, reverse=False):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        Bsp, Csp, Tsp = spect.shape
        # spect_orig = spect.clone().detach()
        # print(spect_orig.shape)

        if pos is not None:
            pos = self.pos_emb(pos)
            pos = self.pos_layer(pos)
            pos = pos.unsqueeze(2)

        if x_prev is not None:
            if not reverse:
                x_prev = (self.prev_cond_layer1(x_prev))
                x_prev = x_prev.squeeze()
            else:
                B, C, T = audio_shape
                prev_conv_weight = self.prev_cond_layer1.weight
                prev_conv_bias = self.prev_cond_layer1.bias
                x_prev = F.conv3d(x_prev, prev_conv_weight, prev_conv_bias, padding=[3, 0, 1])[:,:, :4].squeeze(3)
                x_prev = x_prev.permute(0,2,1,3).contiguous().view(B, -1, T)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels

            if pos is not None:
                pos_in = pos[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                pos_in = None
            
            if x_prev is not None:
                x_prev_in = x_prev[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                x_prev_in = None
        
            acts = fused_add_tanh_sigmoid_multiply5(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset+2*self.n_channels, :], 
                pos_in, 
                x_prev_in,
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)


class WaveNetGlow11(nn.Module):
    def __init__(self, in_channels, cin_channels, di_base, pos_group, n_channels, n_layers, prev_cond=False, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        self.pos_group = pos_group
        if pos_group > 1:
            emb_ch = pos_group
            pos_emb = torch.nn.Embedding(pos_group, emb_ch)
            pos_layer = torch.nn.Linear(emb_ch, 2*n_channels*n_layers)
            self.pos_layer = torch.nn.utils.weight_norm(pos_layer, name='weight')
            self.pos_emb = torch.nn.utils.weight_norm(pos_emb, name='weight')
       
        if prev_cond:
            self.prev_cond_layer1 = torch.nn.Conv3d(1, 2*n_channels*n_layers, [3,4,3], padding=[0, 0, 1])

        for i in range(n_layers):
            dilation = di_base ** i

            padding = int((kernel_size*dilation - dilation)) // 2

            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, pos=None, x_prev=None, reverse=False):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        Bsp, Csp, Tsp = spect.shape
        # spect_orig = spect.clone().detach()
        # print(spect_orig.shape)

        if pos is not None:
            pos = F.relu(self.pos_emb(pos))
            pos = self.pos_layer(pos)
            pos = pos.unsqueeze(2)

        if x_prev is not None:
            if not reverse:
                x_prev = (self.prev_cond_layer1(x_prev))
                x_prev = x_prev.squeeze()
            else:
                B, C, T = audio_shape
                prev_conv_weight = self.prev_cond_layer1.weight
                prev_conv_bias = self.prev_cond_layer1.bias
                x_prev = F.conv3d(x_prev, prev_conv_weight, prev_conv_bias, padding=[3, 0, 1])[:,:, :4].squeeze(3)
                x_prev = x_prev.permute(0,2,1,3).contiguous().view(B, -1, T)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels

            if pos is not None:
                pos_in = pos[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                pos_in = None
            
            if x_prev is not None:
                x_prev_in = x_prev[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                x_prev_in = None
        
            acts = fused_add_tanh_sigmoid_multiply5(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset+2*self.n_channels, :], 
                pos_in, 
                x_prev_in,
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)


class WaveNetGlow10(nn.Module):
    def __init__(self, in_channels, cin_channels, di_base, pos_group, n_channels, n_layers, prev_cond=False, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        self.pos_group = pos_group
        if pos_group > 1:
            emb_ch = pos_group
            pos_emb = torch.nn.Embedding(pos_group, emb_ch)
            pos_layer = torch.nn.Linear(emb_ch, 2*n_channels*n_layers)
            self.pos_layer = torch.nn.utils.weight_norm(pos_layer, name='weight')
            self.pos_emb = torch.nn.utils.weight_norm(pos_emb, name='weight')
       
        if prev_cond:
            self.prev_cond_layer1 = torch.nn.Conv3d(1, n_channels, [3,4,3], padding=[0, 0, 1])
            self.prev_cond_layer2 = torch.nn.Conv3d(n_channels, n_channels, [1, 1, 1], padding=[0, 0, 0])
            self.prev_cond_layer3 = torch.nn.Conv3d(n_channels, 2*n_channels*n_layers, [1, 1, 1], padding=[0, 0, 0])

        for i in range(n_layers):
            dilation = di_base ** i

            padding = int((kernel_size*dilation - dilation)) // 2

            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, pos=None, x_prev=None, reverse=False):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        Bsp, Csp, Tsp = spect.shape
        # spect_orig = spect.clone().detach()
        # print(spect_orig.shape)

        if pos is not None:
            pos = F.relu(self.pos_emb(pos))
            pos = self.pos_layer(pos)
            pos = pos.unsqueeze(2)

        if x_prev is not None:
            if not reverse:
                x_prev = F.relu(self.prev_cond_layer1(x_prev))
                x_prev = F.relu(self.prev_cond_layer2(x_prev))
                x_prev = (self.prev_cond_layer3(x_prev))
                x_prev = x_prev.squeeze()
            else:
                B, C, T = audio_shape
                prev_conv_weight = self.prev_cond_layer1.weight
                prev_conv_bias = self.prev_cond_layer1.bias
                x_prev = F.relu(F.conv3d(x_prev, prev_conv_weight, prev_conv_bias, padding=[3, 0, 1])[:,:, :4])
                x_prev = F.relu(self.prev_cond_layer2(x_prev))
                x_prev = self.prev_cond_layer3(x_prev)
                x_prev = x_prev.squeeze(3)
                x_prev = x_prev.permute(0,2,1,3).contiguous().view(B, -1, T)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels

            if pos is not None:
                pos_in = pos[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                pos_in = None
            
            if x_prev is not None:
                x_prev_in = x_prev[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                x_prev_in = None
        
            acts = fused_add_tanh_sigmoid_multiply5(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset+2*self.n_channels, :], 
                pos_in, 
                x_prev_in,
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)


class WaveNetGlow9(nn.Module):
    def __init__(self, in_channels, cin_channels, di_base, pos_group, n_channels, n_layers, prev_cond=False, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        self.pos_group = pos_group
        if pos_group > 1:
            emb_ch = pos_group // 2
            pos_emb = torch.nn.Embedding(pos_group, emb_ch)
            pos_layer = torch.nn.Linear(emb_ch, 2*n_channels*n_layers)
            self.pos_layer = torch.nn.utils.weight_norm(pos_layer, name='weight')
            self.pos_emb = torch.nn.utils.weight_norm(pos_emb, name='weight')
       
        if prev_cond:
            self.prev_cond_layer1 = torch.nn.Conv3d(1, 2*n_channels*n_layers, [3,4,3], padding=[0, 0, 1])

        for i in range(n_layers):
            dilation = di_base ** i

            padding = int((kernel_size*dilation - dilation)) // 2

            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, pos=None, x_prev=None, reverse=False):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        Bsp, Csp, Tsp = spect.shape
        # spect_orig = spect.clone().detach()
        # print(spect_orig.shape)

        if pos is not None:
            pos = F.relu(self.pos_emb(pos))
            pos = self.pos_layer(pos)
            pos = pos.unsqueeze(2)

        if x_prev is not None:
            if not reverse:
                x_prev = (self.prev_cond_layer1(x_prev))
                x_prev = x_prev.squeeze()
            else:
                B, C, T = audio_shape
                prev_conv_weight = self.prev_cond_layer1.weight
                prev_conv_bias = self.prev_cond_layer1.bias
                x_prev = F.conv3d(x_prev, prev_conv_weight, prev_conv_bias, padding=[3, 0, 1])[:,:, :4].squeeze(3)
                x_prev = x_prev.permute(0,2,1,3).contiguous().view(B, -1, T)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels

            if pos is not None:
                pos_in = pos[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                pos_in = None
            
            if x_prev is not None:
                x_prev_in = x_prev[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                x_prev_in = None
        
            acts = fused_add_tanh_sigmoid_multiply5(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset+2*self.n_channels, :], 
                pos_in, 
                x_prev_in,
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)


class WaveNetGlow15(nn.Module):
    def __init__(self, in_channels, cin_channels, di_base, pos_group, n_channels, n_layers, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group
        self.n_type = 8
        self.in_layers = torch.nn.ModuleList()

        for _ in range(self.n_type):
            self.in_layers.append(torch.nn.ModuleList())

        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        
        self.pos_group = pos_group
        if pos_group > 1:
            pos_emb = torch.nn.Embedding(pos_group, 64)
            pos_layer = torch.nn.Linear(64, 2*n_channels*n_layers)
            self.pos_layer = torch.nn.utils.weight_norm(pos_layer, name='weight')
            self.pos_emb = torch.nn.utils.weight_norm(pos_emb, name='weight')

        for i in range(n_layers):
            for j in range(self.n_type):
                interval = (di_base - 1)/(self.n_type-1)
                dilation = round(interval * j) + 1
                padding = int((kernel_size*dilation - dilation)) // 2
                in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                        dilation=dilation, padding=padding)
                in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')

                self.in_layers[j].append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, pos=None):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        Bsp, Csp, Tsp = spect.shape
        # spect_orig = spect.clone().detach()
        # print(spect_orig.shape)

        if pos is not None:
            pos = F.relu(self.pos_emb(pos))
            pos = self.pos_layer(pos)
            pos = pos.unsqueeze(2)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            spect_in = spect[:, spect_offset:spect_offset+2*self.n_channels, :]
            aud_in = []
            for j in range(self.n_type):
                aud_in.append(self.in_layers[j][i](audio))
            if pos is not None:
                pos_in = pos[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                pos_in = None

            acts = fused_add_tanh_sigmoid_multiply6(
                n_channels_tensor,
                aud_in,
                spect_in, 
                pos_in)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)

class WaveNetGlow14(nn.Module):
    def __init__(self, in_channels, cin_channels, di_base, pos_group, n_channels, n_layers, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group
        self.n_type = 3
        self.in_layers = torch.nn.ModuleList()

        for _ in range(self.n_type):
            self.in_layers.append(torch.nn.ModuleList())

        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        
        self.pos_group = pos_group
        if pos_group > 1:
            pos_emb = torch.nn.Embedding(pos_group, 64)
            pos_layer = torch.nn.Linear(64, 2*n_channels*n_layers)
            self.pos_layer = torch.nn.utils.weight_norm(pos_layer, name='weight')
            self.pos_emb = torch.nn.utils.weight_norm(pos_emb, name='weight')

        for i in range(n_layers):
            for j in range(self.n_type):
                interval = (di_base - 1)/(self.n_type-1)
                dilation = round(interval * j) + 1
                padding = int((kernel_size*dilation - dilation)) // 2
                in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                        dilation=dilation, padding=padding)
                in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')

                self.in_layers[j].append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, pos=None):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        Bsp, Csp, Tsp = spect.shape
        # spect_orig = spect.clone().detach()
        # print(spect_orig.shape)

        if pos is not None:
            pos = F.relu(self.pos_emb(pos))
            pos = self.pos_layer(pos)
            pos = pos.unsqueeze(2)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            spect_in = spect[:, spect_offset:spect_offset+2*self.n_channels, :]
            aud_in = []
            for j in range(self.n_type):
                aud_in.append(self.in_layers[j][i](audio))
            if pos is not None:
                pos_in = pos[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                pos_in = None

            acts = fused_add_tanh_sigmoid_multiply6(
                n_channels_tensor,
                aud_in,
                spect_in, 
                pos_in)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)



class WaveNetGlow13(nn.Module):
    def __init__(self, in_channels, cin_channels, di_base, pos_group, n_channels, n_layers, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group
        self.n_type = 8
        self.in_layers = torch.nn.ModuleList()

        for _ in range(self.n_type):
            self.in_layers.append(torch.nn.ModuleList())

        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        
        self.pos_group = pos_group
        if pos_group > 1:
            pos_emb = torch.nn.Embedding(pos_group, 64)
            pos_layer = torch.nn.Linear(64, 2*n_channels*n_layers)
            self.pos_layer = torch.nn.utils.weight_norm(pos_layer, name='weight')
            self.pos_emb = torch.nn.utils.weight_norm(pos_emb, name='weight')

        for i in range(n_layers):
            for j in range(self.n_type):
                interval = log2(di_base) / (self.n_type - 1)
                # print('interval', interval)
                dilation = round(2 ** (interval * j))
                padding = int((kernel_size*dilation - dilation)) // 2
                in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                        dilation=dilation, padding=padding)
                in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')

                self.in_layers[j].append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, pos=None):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        Bsp, Csp, Tsp = spect.shape
        # spect_orig = spect.clone().detach()
        # print(spect_orig.shape)

        if pos is not None:
            pos = F.relu(self.pos_emb(pos))
            pos = self.pos_layer(pos)
            pos = pos.unsqueeze(2)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            spect_in = spect[:, spect_offset:spect_offset+2*self.n_channels, :]
            aud_in = []
            for j in range(self.n_type):
                aud_in.append(self.in_layers[j][i](audio))
            if pos is not None:
                pos_in = pos[:, spect_offset:spect_offset+2*self.n_channels]
            else:
                pos_in = None

            acts = fused_add_tanh_sigmoid_multiply6(
                n_channels_tensor,
                aud_in,
                spect_in, 
                pos_in)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)

class WaveNetGlow2(nn.Module):
    def __init__(self, in_channels, cin_channels, di_base, pos_group, n_channels, n_layers, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        pos_emb = torch.nn.Embedding(pos_group, 128)
        pos_layer = torch.nn.Linear(128, 2*n_channels*n_layers)
        self.pos_layer = torch.nn.utils.weight_norm(pos_layer, name='weight')
        self.pos_emb = torch.nn.utils.weight_norm(pos_emb, name='weight')

        for i in range(n_layers):
            dilation = di_base ** i
            
            padding = int((kernel_size*dilation - dilation)) // 2

            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, pos):
        audio_shape = audio.shape
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        Bsp, Csp, Tsp = spect.shape
        # spect_orig = spect.clone().detach()
        # print(spect_orig.shape)

        pos = F.relu(self.pos_emb(pos))
        pos = self.pos_layer(pos)
        pos = pos.unsqueeze(2)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            acts = fused_add_tanh_sigmoid_multiply2(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset+2*self.n_channels, :], 
                pos[:, spect_offset:spect_offset+2*self.n_channels], 
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)


def shift_1d(x):
    # shift tensor on height by one for WaveFlowAR modeling
    x = F.pad(x, (0, 0, 1, 0))
    x = x[:, :, :-1, :]
    return x

class WaveFlowCoupling2D(nn.Module):
    def __init__(self, in_channel, cin_channel, filter_size=256, num_layer=6, num_height=None,
                 layers_per_dilation_h_cycle=3):
        super().__init__()
        assert num_height is not None
        self.num_height = num_height
        self.layers_per_dilation_h_cycle = layers_per_dilation_h_cycle
        # dilation for width & height generation loop
        self.dilation_h = []
        self.dilation_w = []
        self.kernel_size = 3
        for i in range(num_layer):
            self.dilation_h.append(2 ** (i % self.layers_per_dilation_h_cycle))
            self.dilation_w.append(2 ** i)
        self.net = Wavenet2D(in_channels=in_channel, out_channels=filter_size,
                             num_layers=num_layer, residual_channels=filter_size,
                             gate_channels=filter_size, skip_channels=filter_size,
                             kernel_size=3, cin_channels=cin_channel, dilation_h=self.dilation_h,
                             dilation_w=self.dilation_w)
        # projector for log_s and t
        self.proj_log_s = ZeroConv2d(filter_size, in_channel)
        self.proj_t = ZeroConv2d(filter_size, in_channel)

    def forward(self, x, c=None):
        x_shift = shift_1d(x)

        feat = self.net(x_shift, c)
        log_s = self.proj_log_s(feat)
        t = self.proj_t(feat)
        out = x * torch.exp(log_s) + t
        
        logdet = torch.sum(log_s)
        return out, logdet

    def reverse(self, z, c=None):
        z_shift = shift_1d(z)

        for i_h in range(self.num_height):
            z_in, c_in = z_shift[:, :, :i_h + 1, :], c[:, :, :i_h + 1, :]
            feat = self.net(z_in, c_in)[:, :, -1, :].unsqueeze(2)
            log_s = self.proj_log_s(feat)
            t = self.proj_t(feat)

            z_trans = z[:, :, i_h, :].unsqueeze(2)
            z[:, :, i_h, :] = ((z_trans - t) * torch.exp(-log_s)).squeeze(2)
            if i_h != (self.num_height - 1):
                z_shift[:, :, i_h + 1] = z[:, :, i_h]
        return z, c


def reverse_order(x):
    # reverse order of x and c along channel dimension
    x = torch.flip(x, dims=(2,))
    return x


class WaveFlow(nn.Module):
    def __init__(self, in_channel, cin_channel, filter_size, num_layer, num_height, layers_per_dilation_h_cycle):
        super().__init__()

        self.coupling = WaveFlowCoupling2D(in_channel, cin_channel, filter_size=filter_size, num_layer=num_layer,
                                           num_height=num_height,
                                           layers_per_dilation_h_cycle=layers_per_dilation_h_cycle, )

    def forward(self, x, c=None):
        out, logdet = self.coupling(x, c)
        out = reverse_order(out)
        c = reverse_order(c)

        return out, c, logdet

    def reverse(self, z, c=None):
        z = reverse_order(z)
        c = reverse_order(c)
        z, c = self.coupling.reverse(z, c)
        return z, c