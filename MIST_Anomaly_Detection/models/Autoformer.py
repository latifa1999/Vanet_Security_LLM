# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.Embed import DataEmbedding, DataEmbedding_wo_pos,DataEmbedding_wo_pos_temp,DataEmbedding_wo_temp
# from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
# from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
# import math
# import numpy as np


# class Model(nn.Module):
#     """
#     Autoformer is the first method to achieve the series-wise connection,
#     with inherent O(LlogL) complexity
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.label_len = configs.label_len
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention

#         # Decomp
#         kernel_size = configs.moving_avg
#         self.decomp = series_decomp(kernel_size)

#         # Embedding
#         # The series-wise connection inherently contains the sequential information.
#         # Thus, we can discard the position embedding of transformers.
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
#             self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)

#         elif configs.embed_type == 3:
#             self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         elif configs.embed_type == 4:
#             self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
        
#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AutoCorrelationLayer(
#                         AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                         output_attention=configs.output_attention),
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
#                         AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
#                                         output_attention=False),
#                         configs.d_model, configs.n_heads),
#                     AutoCorrelationLayer(
#                         AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                         output_attention=False),
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
#         zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
#         seasonal_init, trend_init = self.decomp(x_enc)
#         # decoder input
#         trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
#         seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_pos_temp, DataEmbedding_wo_temp
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer for Classification Task
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = 'classification'
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention if hasattr(configs, 'output_attention') else False

        # Decomp
        kernel_size = configs.moving_avg
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
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=self.output_attention),
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
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        
        if self.output_attention:
            return output, attns
        else:
            return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # Handle case where x_mark_enc is not provided
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4).to(x_enc.device)
        
        # Perform classification
        output = self.classification(x_enc, x_mark_enc)
        
        return output  # [B, num_classes] or ([B, num_classes], attns) if output_attention=True