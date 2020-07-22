import torch
import torch.nn as nn
import torch.nn.functional as F


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


class WaveNet(nn.Module):
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
        print('pos_group', pos_group)
        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        if pos_group > 1:
            pos_emb = torch.nn.Embedding(pos_group, 64)
            pos_layer = torch.nn.Linear(64, 2*n_channels*n_layers)
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

    def forward(self, audio, spect, pos=None):
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