import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    PatchTST for Classification Task
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = 'classification'
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len if hasattr(configs, 'pred_len') else 0
        
        # Use configs values if available, otherwise use defaults
        patch_len = configs.patch_len if hasattr(configs, 'patch_len') else patch_len
        stride = configs.stride if hasattr(configs, 'stride') else stride
        padding = stride

        # Patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Calculate head dimension
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        
        # Classification head
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def classification(self, x_enc, x_mark_enc):
        """
        Classification forward pass
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [bs x nvars x seq_len]
        # enc_out: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # enc_out: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        
        # Reshape: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        
        # Permute: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Classification head
        output = self.flatten(enc_out)  # [bs x nvars x (d_model * patch_num)]
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # [bs x (nvars * d_model * patch_num)]
        output = self.projection(output)  # [bs x num_classes]
        
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass for classification
        x_enc: [B, seq_len, enc_in]
        Returns: [B, num_classes]
        """
        # Handle case where x_mark_enc is not provided
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4).to(x_enc.device)
        
        # Perform classification
        dec_out = self.classification(x_enc, x_mark_enc)
        
        return dec_out  # [B, num_classes]