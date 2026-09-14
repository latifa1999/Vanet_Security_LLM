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

"""
FEDformer for Time Series Classification
Adapted from the forecasting version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_pos_temp, DataEmbedding_wo_temp
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FED_FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np


class Model(nn.Module):
    """
    FEDformer for Time Series Classification
    Performs attention mechanism on frequency domain with O(N) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Version selection
        self.version = getattr(configs, 'version', 'Fourier')  # 'Fourier' or 'Wavelets'
        self.mode_select = getattr(configs, 'mode_select', 'random')
        self.modes = getattr(configs, 'modes', 64)
        
        # Basic parameters
        self.seq_len = configs.seq_len
        self.num_classes = configs.num_classes
        self.output_attention = getattr(configs, 'output_attention', False)
        
        # Model dimensions
        self.d_model = getattr(configs, 'd_model', 512)
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.d_ff = getattr(configs, 'd_ff', 2048)
        self.e_layers = getattr(configs, 'e_layers', 2)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.activation = getattr(configs, 'activation', 'gelu')
        
        # Decomposition
        kernel_size = getattr(configs, 'moving_avg', 25)
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)
        
        # Embedding
        embed_type = getattr(configs, 'embed_type', 0)
        
        if embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_pos(
                configs.enc_in, self.d_model,
                getattr(configs, 'embed', 'timeF'),
                getattr(configs, 'freq', 'h'),
                self.dropout
            )
        elif embed_type == 1:
            self.enc_embedding = DataEmbedding(
                configs.enc_in, self.d_model,
                getattr(configs, 'embed', 'timeF'),
                getattr(configs, 'freq', 'h'),
                self.dropout
            )
        elif embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos_temp(
                configs.enc_in, self.d_model,
                getattr(configs, 'embed', 'timeF'),
                getattr(configs, 'freq', 'h'),
                self.dropout
            )
        elif embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(
                configs.enc_in, self.d_model,
                getattr(configs, 'embed', 'timeF'),
                getattr(configs, 'freq', 'h'),
                self.dropout
            )
        
        # Encoder self-attention: Fourier or Wavelet
        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(
                ich=self.d_model,
                L=getattr(configs, 'L', 1),
                base=getattr(configs, 'base', 'legendre')
            )
        else:  # Fourier
            encoder_self_att = FourierBlock(
                in_channels=self.d_model,
                out_channels=self.d_model,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select
            )
        
        # Encoder
        enc_modes = int(min(self.modes, self.seq_len // 2))
        print(f'FEDformer Classification - enc_modes: {enc_modes}, version: {self.version}')
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        self.d_model,
                        self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    moving_avg=kernel_size,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.num_classes)
        )
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Forward pass for classification
        
        Args:
            x_enc: [Batch, seq_len, enc_in] - input time series
            x_mark_enc: [Batch, seq_len, mark_size] - time encodings (optional)
            
        Returns:
            logits: [Batch, num_classes] - classification logits
        """
        
        # Handle time encodings - create dummy if not provided
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4, 
                                    device=x_enc.device, dtype=x_enc.dtype)
        
        # Series decomposition (optional - can use seasonal, trend, or original)
        # Using seasonal component for classification
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # Embedding (using seasonal component)
        enc_out = self.enc_embedding(seasonal_init, x_mark_enc)
        # enc_out shape: [Batch, seq_len, d_model]
        
        # Encoder with Fourier/Wavelet attention
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # enc_out shape: [Batch, seq_len, d_model]
        
        # Global Average Pooling over time dimension
        # [Batch, seq_len, d_model] -> [Batch, d_model, seq_len]
        enc_out = enc_out.transpose(1, 2)
        enc_out = self.gap(enc_out)  # [Batch, d_model, 1]
        enc_out = enc_out.squeeze(-1)  # [Batch, d_model]
        
        # Classification
        logits = self.classifier(enc_out)  # [Batch, num_classes]
        
        if self.output_attention:
            return logits, attns
        else:
            return logits


class ModelWithBothComponents(nn.Module):
    """
    Enhanced FEDformer for Classification
    Uses both trend and seasonal components explicitly
    """
    def __init__(self, configs):
        super(ModelWithBothComponents, self).__init__()
        
        # Version selection
        self.version = getattr(configs, 'version', 'Fourier')
        self.mode_select = getattr(configs, 'mode_select', 'random')
        self.modes = getattr(configs, 'modes', 64)
        
        self.seq_len = configs.seq_len
        self.num_classes = configs.num_classes
        self.output_attention = getattr(configs, 'output_attention', False)
        
        # Model dimensions
        self.d_model = getattr(configs, 'd_model', 512)
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.d_ff = getattr(configs, 'd_ff', 2048)
        self.e_layers = getattr(configs, 'e_layers', 2)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.activation = getattr(configs, 'activation', 'gelu')
        
        # Decomposition
        kernel_size = getattr(configs, 'moving_avg', 25)
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)
        
        # Embedding for seasonal component
        embed_type = getattr(configs, 'embed_type', 0)
        
        if embed_type == 0:
            self.seasonal_embedding = DataEmbedding_wo_pos(
                configs.enc_in, self.d_model,
                getattr(configs, 'embed', 'timeF'),
                getattr(configs, 'freq', 'h'),
                self.dropout
            )
        else:
            self.seasonal_embedding = DataEmbedding(
                configs.enc_in, self.d_model,
                getattr(configs, 'embed', 'timeF'),
                getattr(configs, 'freq', 'h'),
                self.dropout
            )
        
        # Trend projection (simpler processing)
        self.trend_projection = nn.Linear(configs.enc_in, self.d_model)
        
        # Encoder self-attention
        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(
                ich=self.d_model,
                L=getattr(configs, 'L', 1),
                base=getattr(configs, 'base', 'legendre')
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=self.d_model,
                out_channels=self.d_model,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select
            )
        
        # Seasonal encoder
        self.seasonal_encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        self.d_model,
                        self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    moving_avg=kernel_size,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )
        
        # Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.num_classes)
        )
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Forward with both seasonal and trend components
        """
        
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4,
                                    device=x_enc.device, dtype=x_enc.dtype)
        
        # Decompose into seasonal and trend
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # Process seasonal component with Fourier/Wavelet attention
        seasonal_enc = self.seasonal_embedding(seasonal_init, x_mark_enc)
        seasonal_enc, _ = self.seasonal_encoder(seasonal_enc, attn_mask=enc_self_mask)
        seasonal_enc = seasonal_enc.transpose(1, 2)
        seasonal_enc = self.gap(seasonal_enc).squeeze(-1)  # [Batch, d_model]
        
        # Process trend component (simpler processing)
        trend_enc = self.trend_projection(trend_init)  # [Batch, seq_len, d_model]
        trend_enc = trend_enc.transpose(1, 2)
        trend_enc = self.gap(trend_enc).squeeze(-1)  # [Batch, d_model]
        
        # Fuse seasonal and trend
        fused = torch.cat([seasonal_enc, trend_enc], dim=-1)  # [Batch, d_model * 2]
        fused = self.fusion(fused)  # [Batch, d_model]
        
        # Classify
        logits = self.classifier(fused)  # [Batch, num_classes]
        
        return logits