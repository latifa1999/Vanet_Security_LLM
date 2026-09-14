# import sys
# import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
# from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
# from layers.FourierCorrelation import SpectralConv1d, SpectralConvCross1d, SpectralConv1d_local, \
#     SpectralConvCross1d_local
# from layers.mwt import MWT_CZ1d_cross, mwt_transform
# from layers.SelfAttention_Family import FullAttention, ProbAttention
# from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, \
#     series_decomp_multi
# import math
# import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class Model(nn.Module):
#     """
#     Autoformer is the first method to achieve the series-wise connection,
#     with inherent O(LlogL) complexity
#     """

#     def __init__(self, configs):
#         super(Model, self).__init__()
#         # self.modes = configs.modes
#         self.seq_len = configs.seq_len
#         self.label_len = configs.label_len
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention

#         # Decomp
#         kernel_size = configs.moving_avg
#         # self.decomp = series_decomp(kernel_size)
#         kernel_size = [kernel_size]
#         self.decomp = series_decomp_multi(kernel_size)

#         # Embedding
#         # The series-wise connection inherently contains the sequential information.
#         # Thus, we can discard the position embedding of transformers.
#         self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                   configs.dropout)
#         self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                   configs.dropout)

#         configs.ab = 2

#         if configs.ab == 0:
#             encoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
#                                               seq_len=self.seq_len, modes1=configs.modes1)
#             decoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
#                                               seq_len=self.seq_len // 2 + self.pred_len, modes1=configs.modes1)
#             decoder_cross_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
#                                                     seq_len_q=self.seq_len // 2 + self.pred_len,
#                                                     seq_len_kv=self.seq_len, modes1=configs.modes1)
#         elif configs.ab == 1:
#             encoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
#                                               seq_len=self.seq_len)
#             decoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
#                                               seq_len=self.seq_len // 2 + self.pred_len)
#             decoder_cross_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                                 output_attention=False, configs=configs)
#         elif configs.ab == 2:
#             encoder_self_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                                output_attention=configs.output_attention)
#             decoder_self_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                                output_attention=configs.output_attention)
#             decoder_cross_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
#                                                     seq_len_q=self.seq_len // 2 + self.pred_len,
#                                                     seq_len_kv=self.seq_len)
#         elif configs.ab == 3:
#             encoder_self_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                                output_attention=configs.output_attention, configs=configs)
#             decoder_self_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                                output_attention=configs.output_attention, configs=configs)
#             decoder_cross_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                                 output_attention=False, configs=configs)
#         elif configs.ab == 4:
#             encoder_self_att = FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                              output_attention=configs.output_attention)
#             decoder_self_att = FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                              output_attention=configs.output_attention)
#             decoder_cross_att = FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                               output_attention=configs.output_attention)
#         elif configs.ab == 8:
#             encoder_self_att = ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                              output_attention=configs.output_attention)
#             decoder_self_att = ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                              output_attention=configs.output_attention)
#             decoder_cross_att = ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                               output_attention=configs.output_attention)
#         elif configs.ab == 5:
#             encoder_self_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
#                                                    seq_len_q=self.seq_len, seq_len_kv=self.seq_len)
#             decoder_self_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
#                                                    seq_len_q=self.seq_len // 2 + self.pred_len,
#                                                    seq_len_kv=self.seq_len // 2 + self.pred_len)
#             decoder_cross_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
#                                                     seq_len_q=self.seq_len // 2 + self.pred_len,
#                                                     seq_len_kv=self.seq_len)
#         elif configs.ab == 6:
#             encoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
#                                               seq_len=self.seq_len, modes1=configs.modes1)
#             decoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
#                                               seq_len=self.seq_len // 2 + self.pred_len, modes1=configs.modes1)
#             decoder_cross_att = SpectralConvCross1d_local(in_channels=configs.d_model, out_channels=configs.d_model,
#                                                           seq_len_q=self.seq_len // 2 + self.pred_len,
#                                                           seq_len_kv=self.seq_len, modes1=configs.modes1)
#         elif configs.ab == 7:
#             # encoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model, seq_len=self.seq_len, modes1=configs.modes1)
#             # decoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model, seq_len=self.seq_len//2+self.pred_len, modes1=configs.modes1)
#             # encoder_self_att = mwt_transform(ich=configs.d_model, L=3, alpha=int(self.pred_len/2+1))
#             # decoder_self_att = mwt_transform(ich=configs.d_model, L=3, alpha=int(self.pred_len/2+1))
#             encoder_self_att = mwt_transform(ich=configs.d_model, L=configs.L, base=configs.base)
#             decoder_self_att = mwt_transform(ich=configs.d_model, L=configs.L, base=configs.base)
#             # decoder_cross_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,seq_len_q=self.seq_len//2+self.pred_len, seq_len_kv=self.seq_len, modes1=configs.modes1)
#             decoder_cross_att = MWT_CZ1d_cross(in_channels=configs.d_model, out_channels=configs.d_model,
#                                                seq_len_q=self.seq_len // 2 + self.pred_len, seq_len_kv=self.seq_len,
#                                                modes1=configs.modes1, ich=configs.d_model, base=configs.base,
#                                                activation=configs.cross_activation)
#         elif config.ab == 8:
#             encoder_self_att = SpectralConv1d_local(in_channels=configs.d_model, out_channels=configs.d_model,
#                                                     seq_len=self.seq_len, modes1=configs.modes1)
#             decoder_self_att = SpectralConv1d_local(in_channels=configs.d_model, out_channels=configs.d_model,
#                                                     seq_len=self.seq_len // 2 + self.pred_len, modes1=configs.modes1)
#             decoder_cross_att = SpectralConvCross1d_local(in_channels=configs.d_model, out_channels=configs.d_model,
#                                                           seq_len_q=self.seq_len // 2 + self.pred_len,
#                                                           seq_len_kv=self.seq_len, modes1=configs.modes1)

#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AutoCorrelationLayer(
#                         encoder_self_att,
#                         configs.d_model, configs.n_heads),

#                     configs.d_model,
#                     configs.d_ff,
#                     moving_avg=configs.moving_avg,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=my_Layernorm(configs.d_model)
#         )
#         # Decoder
#         self.decoder = Decoder(
#             [
#                 DecoderLayer(
#                     AutoCorrelationLayer(
#                         decoder_self_att,
#                         configs.d_model, configs.n_heads),
#                     AutoCorrelationLayer(
#                         decoder_cross_att,
#                         configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.c_out,
#                     configs.d_ff,
#                     moving_avg=configs.moving_avg,
#                     dropout=configs.dropout,
#                     activation=configs.activation,
#                 )
#                 for l in range(configs.d_layers)
#             ],
#             norm_layer=my_Layernorm(configs.d_model),
#             projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
#         )
#         a = 2

#     def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
#                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
#         x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4).to(x_enc.device)
#         x_dec = torch.zeros(x_enc.shape[0], 48+720, x_enc.shape[2]).to(x_enc.device)
#         x_mark_dec = torch.zeros(x_enc.shape[0], 48+720, 4).to(x_enc.device)
#         # decomp init
#         mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
#         zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
#         seasonal_init, trend_init = self.decomp(x_enc)
#         # decoder input
#         trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
#         seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
#         # seasonal_init1 = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
#         # enc
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
#         # dec
#         dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
#         seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
#                                                  trend=trend_init)
#         # final
#         dec_out = trend_part + seasonal_part

#         if self.output_attention:
#             return dec_out[:, -self.pred_len:, :], attns
#         else:
#             return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# if __name__ == '__main__':
#     class Configs(object):
#         ab = 0
#         modes1 = 32
#         seq_len = 336
#         label_len = 48
#         pred_len = 720
#         output_attention = True
#         enc_in = 7
#         dec_in = 7
#         d_model = 16
#         embed = 'timeF'
#         dropout = 0.05
#         freq = 'h'
#         factor = 1
#         n_heads = 8
#         d_ff = 16
#         e_layers = 2
#         d_layers = 1
#         moving_avg = [25]
#         c_out = 7
#         activation = 'gelu'
#         wavelet = 0


#     configs = Configs()
#     model = Model(configs)

#     enc = torch.randn([32, configs.seq_len, 7])
#     enc_mark = torch.randn([32, configs.seq_len, 4])

#     dec = torch.randn([32, configs.label_len + configs.pred_len, 7])
#     dec_mark = torch.randn([32, configs.label_len + configs.pred_len, 4])
#     out = model.forward(enc, enc_mark, dec, dec_mark)
#     print('input shape', enc.shape)
#     print('output shape', out[0].shape)
#     a = 1


#     def count_parameters(model):
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)


#     print('model size', count_parameters(model) / (1024 * 1024))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy import special as ss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def transition(N):
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1)[:, None]
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1, (-1.) ** (i - j + 1)) * R
    B = (-1.) ** Q[:, None] * R
    return A, B


class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super(HiPPO_LegT, self).__init__()
        self.N = N
        A, B = transition(N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A).to(device))
        self.register_buffer('B', torch.Tensor(B).to(device))
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer('eval_matrix', torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T).to(device))

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(device)
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            new = f @ self.B.unsqueeze(0)
            c = F.linear(c, self.A) + new
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, ratio=0.5):
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.modes = min(32, seq_len // 2)
        self.index = list(range(0, self.modes))

        self.scale = (1 / (in_channels * out_channels))
        self.weights_real = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))
        self.weights_imag = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))

    def compl_mul1d(self, order, x, weights_real, weights_imag):
        return torch.complex(
            torch.einsum(order, x.real, weights_real) - torch.einsum(order, x.imag, weights_imag),
            torch.einsum(order, x.real, weights_imag) + torch.einsum(order, x.imag, weights_real)
        )

    def forward(self, x):
        B, H, E, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, H, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        a = x_ft[:, :, :, :self.modes]
        out_ft[:, :, :, :self.modes] = self.compl_mul1d("bjix,iox->bjox", a, self.weights_real, self.weights_imag)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class Model(nn.Module):
    """
    FILM (Frequency Improved Legendre Memory) for Classification Task
    Paper link: https://arxiv.org/abs/2205.08897
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = 'classification'
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len  # For classification, use same length

        self.layers = configs.e_layers
        self.enc_in = configs.enc_in
        self.e_layers = configs.e_layers
        
        # Affine transformation parameters
        self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

        # Multiscale and window configuration
        self.multiscale = [1, 2, 4]
        self.window_size = [256]
        configs.ratio = 0.5 if not hasattr(configs, 'ratio') else configs.ratio
        
        # HiPPO-LegT projections for multiple scales
        self.legts = nn.ModuleList(
            [HiPPO_LegT(N=n, dt=1. / self.pred_len / i) 
             for n in self.window_size 
             for i in self.multiscale]
        )
        
        # Spectral convolution layers
        self.spec_conv_1 = nn.ModuleList(
            [SpectralConv1d(
                in_channels=n, 
                out_channels=n,
                seq_len=min(self.pred_len, self.seq_len),
                ratio=configs.ratio
            ) for n in self.window_size 
              for _ in range(len(self.multiscale))]
        )
        
        # MLP to aggregate multiscale features
        self.mlp = nn.Linear(len(self.multiscale) * len(self.window_size), 1)

        # Classification head
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

    def classification(self, x_enc, x_mark_enc):
        """
        Classification forward pass using FILM architecture
        """
        # Apply affine transformation
        x_enc = x_enc * self.affine_weight + self.affine_bias
        
        x_decs = []
        jump_dist = 0
        
        # Process through multiscale spectral convolutions
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            # Get input length for this scale
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.pred_len
            x_in = x_enc[:, -x_in_len:]
            
            # Apply HiPPO-LegT projection
            legt = self.legts[i]
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
            
            # Apply spectral convolution
            out1 = self.spec_conv_1[i](x_in_c)
            
            # Extract features
            if self.seq_len >= self.pred_len:
                x_dec_c = out1.transpose(2, 3)[:, :, self.pred_len - 1 - jump_dist, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            
            # Reconstruct using Legendre basis
            x_dec = x_dec_c @ legt.eval_matrix[-self.pred_len:, :].T
            x_decs.append(x_dec)
        
        # Aggregate multiscale features
        x_dec = torch.stack(x_decs, dim=-1)
        x_dec = self.mlp(x_dec).squeeze(-1).permute(0, 2, 1)

        # Classification head
        output = self.act(x_dec)
        output = self.dropout(output)
        
        # Flatten: (batch_size, seq_length * enc_in)
        output = output.reshape(output.shape[0], -1)
        
        # Project to number of classes
        output = self.projection(output)
        
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass for classification
        x_enc: [B, seq_len, enc_in]
        Returns: [B, num_class]
        """
        # Handle case where x_mark_enc is not provided
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4).to(x_enc.device)
        
        # Perform classification
        dec_out = self.classification(x_enc, x_mark_enc)
        
        return dec_out  # [B, num_class]


# if __name__ == '__main__':
#     class Configs(object):
#         task_name = 'classification'
#         seq_len = 96
#         pred_len = 0  # Will be set to seq_len for classification
#         enc_in = 7
#         e_layers = 2
#         dropout = 0.05
#         ratio = 0.5
#         num_class = 2

#     configs = Configs()
#     model = Model(configs)

#     print('='*80)
#     print('FILM Classification Model Test')
#     print('='*80)
#     print(f'Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
#     print(f'Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad) / (1024 * 1024):.2f} MB')
    
#     # Test forward pass
#     batch_size = 32
#     enc = torch.randn([batch_size, configs.seq_len, configs.enc_in])
#     enc_mark = torch.randn([batch_size, configs.seq_len, 4])

#     out = model.forward(enc, enc_mark)
    
#     print(f'\nInput shape:  {enc.shape}')
#     print(f'Output shape: {out.shape}')
#     print(f'Expected:     [{batch_size}, {configs.num_class}]')
#     print('='*80)
    
#     assert out.shape == (batch_size, configs.num_class), "Output shape mismatch!"
#     print('✓ Test passed!')