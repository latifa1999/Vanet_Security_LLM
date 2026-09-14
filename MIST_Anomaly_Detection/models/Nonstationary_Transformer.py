import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class Model(nn.Module):
    """
    Non-Stationary Transformer for Classification Task
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = 'classification'
        self.pred_len = configs.pred_len if hasattr(configs, 'pred_len') else 0
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len if hasattr(configs, 'label_len') else 0

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder with De-stationary Attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=False), configs.d_model, configs.n_heads),
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

        # De-stationary learners (KEY INNOVATION!)
        # p_hidden_dims and p_hidden_layers should be in configs
        p_hidden_dims = configs.p_hidden_dims if hasattr(configs, 'p_hidden_dims') else [128, 128]
        p_hidden_layers = configs.p_hidden_layers if hasattr(configs, 'p_hidden_layers') else 2
        
        # Tau learner: learns variance scaling factor
        self.tau_learner = Projector(
            enc_in=configs.enc_in, 
            seq_len=configs.seq_len, 
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers, 
            output_dim=1
        )
        
        # Delta learner: learns mean shift
        self.delta_learner = Projector(
            enc_in=configs.enc_in, 
            seq_len=configs.seq_len,
            hidden_dims=p_hidden_dims, 
            hidden_layers=p_hidden_layers,
            output_dim=configs.seq_len
        )

    def classification(self, x_enc, x_mark_enc):
        """
        Classification forward pass with de-stationary attention
        """
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        std_enc = torch.sqrt(
            torch.var(x_enc - mean_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # B x 1 x E
        
        # Learn de-stationary factors
        # Tau: variance scaling factor (B x 1, positive scalar)
        tau = self.tau_learner(x_raw, std_enc).exp()
        
        # Delta: mean shift (B x S)
        delta = self.delta_learner(x_raw, mean_enc)
        
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Encoder with de-stationary attention
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # The output transformer encoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        
        # Flatten: (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        
        # Project to num_classes: (batch_size, num_classes)
        output = self.projection(output)
        
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass for classification
        x_enc: [B, seq_len, enc_in]
        x_mark_enc: [B, seq_len, temporal_features] (optional, not used in classification)
        Returns: [B, num_classes]
        """
        # Handle case where x_mark_enc is not provided
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4).to(x_enc.device)
        
        # Perform classification
        dec_out = self.classification(x_enc, x_mark_enc)
        
        return dec_out  # [B, num_classes]