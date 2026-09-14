# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.Embed import DataEmbedding, DataEmbedding_wo_pos,DataEmbedding_wo_pos_temp,DataEmbedding_wo_temp
# from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
# from layers.FED_FourierCorrelation import FourierBlock, FourierCrossAttention
# from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
# from layers.SelfAttention_Family import FullAttention, ProbAttention
# # from layers.FED_wo_decomp import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
# from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
# import math
# import numpy as np


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class Model(nn.Module):
#     """
#     FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.version = 'Fourier'
#         self.mode_select = 'random'
#         self.modes = 64
#         self.seq_len = configs.seq_len
#         self.label_len = configs.label_len
#         self.pred_len = configs.pred_len
#         self.output_attention = False

#         # Decomp
#         kernel_size = configs.moving_avg
#         if isinstance(kernel_size, list):
#             self.decomp = series_decomp_multi(kernel_size)
#         else:
#             self.decomp = series_decomp(kernel_size)

#         # Embedding
#         # The series-wise connection inherently contains the sequential information.
#         # Thus, we can discard the position embedding of transformers.
#         # self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#         #                                           configs.dropout)
#         # self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#         #                                           configs.dropout)
#         if configs.embed_type == 0:
#             self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         elif configs.embed_type == 1:
#             self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         elif configs.embed_type == 2:
#             self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         elif configs.embed_type == 3:
#             self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)

#         if self.version == 'Wavelets':
#             encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
#             decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
#             decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
#                                                   out_channels=configs.d_model,
#                                                   seq_len_q=self.seq_len // 2 + self.pred_len,
#                                                   seq_len_kv=self.seq_len,
#                                                   modes=self.modes,
#                                                   ich=configs.d_model,
#                                                   base=configs.base,
#                                                   activation=configs.cross_activation)
#         else:
#             encoder_self_att = FourierBlock(in_channels=configs.d_model,
#                                             out_channels=configs.d_model,
#                                             seq_len=self.seq_len,
#                                             modes=self.modes,
#                                             mode_select_method=self.mode_select)
#             decoder_self_att = FourierBlock(in_channels=configs.d_model,
#                                             out_channels=configs.d_model,
#                                             seq_len=self.seq_len//2+self.pred_len,
#                                             modes=self.modes,
#                                             mode_select_method=self.mode_select)
#             decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
#                                                       out_channels=configs.d_model,
#                                                       seq_len_q=self.seq_len//2+self.pred_len,
#                                                       seq_len_kv=self.seq_len,
#                                                       modes=self.modes,
#                                                       mode_select_method=self.mode_select)
#         # Encoder
#         enc_modes = int(min(self.modes, configs.seq_len//2))
#         dec_modes = int(min(self.modes, (configs.seq_len//2+configs.pred_len)//2))
#         print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

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
#         modes = 32
#         mode_select = 'random'
#         # version = 'Fourier'
#         version = 'Wavelets'
#         moving_avg = [12, 24]
#         L = 1
#         base = 'legendre'
#         cross_activation = 'tanh'
#         seq_len = 96
#         label_len = 48
#         pred_len = 96
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
#         c_out = 7
#         activation = 'gelu'
#         wavelet = 0

#     configs = Configs()
#     model = Model(configs)

#     print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
#     enc = torch.randn([3, configs.seq_len, 7])
#     enc_mark = torch.randn([3, configs.seq_len, 4])

#     dec = torch.randn([3, configs.seq_len//2+configs.pred_len, 7])
#     dec_mark = torch.randn([3, configs.seq_len//2+configs.pred_len, 4])
#     out = model.forward(enc, enc_mark, dec, dec_mark)
#     print(out)

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_pos_temp, DataEmbedding_wo_temp
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np


class Model(nn.Module):
    """
    FEDformer for Classification Task
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """
    def __init__(self, configs, version='Fourier', mode_select='random', modes=64):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(Model, self).__init__()
        self.task_name = 'classification'
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention if hasattr(configs, 'output_attention') else False

        # Version and mode settings
        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        # Attention mechanism based on version
        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, 
                L=configs.L if hasattr(configs, 'L') else 1, 
                base=configs.base if hasattr(configs, 'base') else 'legendre'
            )
        else:  # Fourier
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select
            )

        # Encoder
        enc_modes = int(min(self.modes, configs.seq_len // 2))
        print('enc_modes: {}'.format(enc_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # Classification head
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def classification(self, x_enc, x_mark_enc):
        """
        Classification forward pass
        """
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # The output transformer encoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        
        # Flatten: (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        
        # Project to num_classes: (batch_size, num_classes)
        output = self.projection(output)
        
        if self.output_attention:
            return output, attns
        else:
            return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Forward pass for classification
        x_enc: [B, seq_len, channels]
        x_mark_enc: [B, seq_len, temporal_features] (optional)
        Returns: [B, num_classes] or ([B, num_classes], attns) if output_attention=True
        """
        # Handle case where x_mark_enc is not provided
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4).to(x_enc.device)

        # Perform classification
        output = self.classification(x_enc, x_mark_enc)

        return output  # [B, num_classes] or ([B, num_classes], attns)