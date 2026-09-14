import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer for Classification Task
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = 'classification'
        self.pred_len = configs.pred_len if hasattr(configs, 'pred_len') else 0
        self.output_attention = configs.output_attention if hasattr(configs, 'output_attention') else False
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.num_class = configs.num_class
        
        # Embedding - Support multiple embedding types
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
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
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=self.output_attention), 
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
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

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass for classification
        x_enc: [B, seq_len, enc_in]
        x_mark_enc: [B, seq_len, temporal_features] (optional)
        Returns: [B, num_classes] or ([B, num_classes], attns) if output_attention=True
        """
        # Handle case where x_mark_enc is not provided
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4).to(x_enc.device)
        
        # Perform classification
        dec_out = self.classification(x_enc, x_mark_enc)
        
        return dec_out  # [B, num_classes]